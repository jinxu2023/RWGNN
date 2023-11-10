import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense

from common_functions.GNN_layers import GCN_Layer
from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class NIMCGCN_Model(Model):
    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, optimizer='adam',
                 seed=0, hyper_paras=(1,), **kwargs):
        super(NIMCGCN_Model, self).__init__()
        self.units = units
        self.seed = seed
        self.levels = levels
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.hyper_paras = hyper_paras

        self.init_data(input_data)

        self.GCN_d = [GCN_Layer(units=units,
                                activation=tf.nn.relu,
                                seed=seed)
                      for i in range(self.levels)]
        self.GCN_t = [GCN_Layer(units=units,
                                activation=tf.nn.relu,
                                seed=seed)
                      for i in range(self.levels)]
        self.dense_d = Dense(units=units,
                             activation=tf.nn.relu,
                             use_bias=False)
        self.dense_t = Dense(units=units,
                             activation=tf.nn.relu,
                             use_bias=False)

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
        self.D = tf.convert_to_tensor(D, dtype='float32')
        self.T = tf.convert_to_tensor(T, dtype='float32')
        self.R_train = tf.convert_to_tensor(R_train, dtype='float32')
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')

        A = np.r_[np.c_[D, R_train], np.c_[R_train.T, T]]
        self.A = tf.convert_to_tensor(A, dtype='float32')
        # self.R = R
        self.d_num = D.shape[0]
        self.t_num = T.shape[0]
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

    def call(self, inputs, training=None, mask=None):
        H_d, H_t = inputs[0][0], inputs[1][0]
        for i in range(self.levels):
            H_d = self.GCN_d[i]([self.D, H_d])
            H_t = self.GCN_t[i]([self.T, H_t])
        H_d = self.dense_d(H_d)
        H_t = self.dense_t(H_t)
        R_pred = H_d @ H_t.T
        loss = self.calc_loss(R_pred, self.R_truth)
        self.add_loss(loss)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d, axis=0),
                tf.expand_dims(H_t, axis=0)]

    def get_config(self):
        config = {'units': self.units,
                  'seed': self.seed,
                  'levels': self.levels,
                  'optimizer': self._optimizer,
                  'epochs': self.epochs
                  }

        return config

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

    def calc_loss(self, X_pred, X):
        loss_LP_pos = tf.reduce_mean(tf.square(
            tf.cast(X == 1, 'float32') * (X_pred - X)))
        loss_LP_neg = tf.reduce_mean(tf.square(
            tf.cast(X == 0, 'float32') * (X_pred - X)))
        loss_LP = loss_LP_pos + 0.2 * loss_LP_neg

        return loss_LP
