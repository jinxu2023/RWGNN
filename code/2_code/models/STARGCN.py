import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform

from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class STARGCN_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, optimizer='adam',
                 seed=0, loss_mode='RDT', hyper_paras=(1,), **kwargs):
        super(STARGCN_Model, self).__init__()
        self.levels = levels
        self.units = units
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.loss_mode = loss_mode
        self.hyper_paras = hyper_paras
        self.kwargs = kwargs

        self.init_data(input_data)

        self.GCMC_Layers = [GCMC_Layer(units=units,
                                       seed=seed)
                            for i in range(levels)]

        self.dense_output_d = Dense(units=self.units,
                                    activation=tf.nn.relu,
                                    use_bias=True,
                                    kernel_initializer=GlorotUniform(self.seed))
        self.dense_output_t = Dense(units=self.units,
                                    activation=tf.nn.relu,
                                    use_bias=True,
                                    kernel_initializer=GlorotUniform(self.seed))
        self.dense_decoder_d1 = Dense(units=self.units,
                                      activation=tf.nn.relu,
                                      use_bias=False,
                                      kernel_initializer=GlorotUniform(self.seed))
        self.dense_decoder_d2 = Dense(units=self.units,
                                      activation=None,
                                      use_bias=False,
                                      kernel_initializer=GlorotUniform(self.seed))
        self.dense_decoder_t1 = Dense(units=self.units,
                                      activation=tf.nn.relu,
                                      use_bias=False,
                                      kernel_initializer=GlorotUniform(self.seed))
        self.dense_decoder_t2 = Dense(units=self.units,
                                      activation=None,
                                      use_bias=False,
                                      kernel_initializer=GlorotUniform(self.seed))

        if Configs.metric_config['metric_group_idx'] == 0:
            self.es_callback = EarlyStopping(monitor=Configs.metric_config['cv_metric'],
                                             patience=Configs.metric_config['patience'],
                                             mode='max', restore_best_weights=True)
        else:
            self.es_callback = EarlyStopping(monitor='loss',
                                             patience=Configs.metric_config['patience'],
                                             mode='min', restore_best_weights=True)

        self.compile(optimizer=self._optimizer)

    def init_data(self, input_data):
        [D, T, R_train, R_truth, self.H_d, self.H_t, mask] = input_data[:7]
        self.R = tf.convert_to_tensor(R_train, dtype='float32')
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')

        self.d_num = D.shape[0]
        self.t_num = T.shape[0]
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

    def call(self, inputs, training=None, mask=None):
        H_d, H_t = inputs[0][0], inputs[1][0]
        H_d_dec, H_t_dec = H_d, H_t
        loss = 0
        for i in range(self.levels):
            H_d_enc, H_t_enc = self.GCMC_Layers[i]([self.R, H_d_dec, H_t_dec])
            # d_output_features = self.dense_output_d(d_new_features)
            # t_output_features = self.dense_output_t(t_new_features)
            R_pred = H_d_enc @ H_t_enc.T
            loss += self.MSE_loss(R_pred, self.R)

            H_d_dec = self.dense_decoder_d2(self.dense_decoder_d1(H_d_enc))
            H_t_dec = self.dense_decoder_t2(self.dense_decoder_t1(H_t_enc))
            loss += self.hyper_paras[0] * tf.reduce_sum((H_d_dec - H_d_enc) ** 2) / self.d_num
            loss += self.hyper_paras[0] * tf.reduce_sum((H_t_dec - H_t_enc) ** 2) / self.d_num
        self.add_loss(loss)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d_enc, axis=0),
                tf.expand_dims(H_t_enc, axis=0)]

    def get_config(self):
        config = {
            'levels': self.levels,
            'units': self.units,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'seed': self.seed
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        H_d = tf.expand_dims(self.H_d, axis=0)
        H_t = tf.expand_dims(self.H_t, axis=0)
        Model.fit(self, x=[H_d, H_t], batch_size=1, epochs=self.epochs, verbose=1,
                  callbacks=[self.es_callback])
        [R_pred, H_d_out, H_t_out] = Model.predict(self, x=[H_d, H_t], batch_size=1, verbose=2)
        self.R_pred = np.squeeze(R_pred)
        self.R_pred[np.isnan(self.R_pred)] = 0
        self.R_pred[np.isinf(self.R_pred)] = 0
        self.H_d_out = np.squeeze(H_d_out)
        self.H_d_out[np.isnan(self.H_d_out)] = 0
        self.H_d_out[np.isinf(self.H_d_out)] = 0
        self.H_t_out = np.squeeze(H_t_out)
        self.H_t_out[np.isnan(self.H_t_out)] = 0
        self.H_t_out[np.isinf(self.H_t_out)] = 0

    def predict(self, **kwargs):
        return [self.R_pred, self.H_d_out, self.H_t_out, None]

    def MSE_loss(self, X_pred, X):
        loss_LP_pos = tf.reduce_mean(tf.square(
            tf.cast(X == 1, 'float32') * (X_pred - X)))
        loss_LP_neg = tf.reduce_mean(tf.square(
            tf.cast(X == 0, 'float32') * (X_pred - X)))
        loss_LP = loss_LP_pos + 0.2 * loss_LP_neg

        return loss_LP


class GCMC_Layer(Layer):
    def __init__(self, units=200, seed=1024):
        super(GCMC_Layer, self).__init__()
        self.units = units
        self.seed = seed

    def build(self, input_shapes):
        self.dense_d = Dense(name='dense_d',
                             units=self.units,
                             activation=None,
                             use_bias=False,
                             kernel_initializer=GlorotUniform(self.seed))
        self.dense_t = Dense(name='dense_t',
                             units=self.units,
                             activation=None,
                             use_bias=False,
                             kernel_initializer=GlorotUniform(self.seed))
        self.dense_output_d = Dense(name='dense_output_d',
                                    units=self.units,
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=GlorotUniform(self.seed))
        self.dense_output_t = Dense(name='dense_output_t',
                                    units=self.units,
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=GlorotUniform(self.seed))

    def call(self, inputs, **kwargs):
        R, H_d, H_t = inputs
        n_neigh_dt = tf.reduce_sum(R, axis=1, keepdims=True)
        n_neigh_td = tf.reduce_sum(R, axis=0, keepdims=True)
        n_neigh_dt = n_neigh_dt + (n_neigh_dt == 0).astype('float32')
        n_neigh_td = n_neigh_td + (n_neigh_td == 0).astype('float32')
        n_neigh_dt_norm = n_neigh_dt ** -0.5
        n_neigh_td_norm = n_neigh_td ** -0.5
        R_norm = n_neigh_dt_norm * R * n_neigh_td_norm

        H_d_out = R_norm @ self.dense_t(H_t)
        H_t_out = R_norm.T @ self.dense_d(H_d)
        d_output_features = self.dense_output_d(tf.nn.relu(H_d_out))
        t_output_features = self.dense_output_t(tf.nn.relu(H_t_out))
        return d_output_features, t_output_features
