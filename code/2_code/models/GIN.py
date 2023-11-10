import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.models import Model

from common_functions.GNN_layers import GIN_Layer
from common_functions.metrics import MetricUtils
from common_functions.configs import Configs


class GIN_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, optimizer='adam', loss_mode='RDT',
                 seed=0, **kwargs):
        super(GIN_Model, self).__init__()
        self.levels = levels
        self.units = units
        self.epochs = epochs
        self._optimizer = optimizer
        self.loss_mode = loss_mode
        self.seed = seed
        self.kwargs = kwargs

        self.init_data(input_data)
        self.GIN_layers = [GIN_Layer(units=units,
                                     seed=seed)
                           for i in range(levels)]
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
        [D, T, R_train, R_truth, H_d, H_t, mask] = input_data[:7]
        self.R = tf.convert_to_tensor(R_train, dtype='float32')
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')

        A = np.r_[np.c_[D, R_train], np.c_[R_train.T, T]]
        self.A = tf.convert_to_tensor(A, dtype='float32')
        self.H = np.r_[H_d, H_t]

        self.d_num, self.t_num = R_train.shape
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

    def call(self, inputs, training=None, mask=None):
        H = inputs[0][0]
        for i in range(self.levels):
            H = self.GIN_layers[i]([self.A, H])

        H_d_out = H[:self.d_num, :]
        H_t_out = H[self.d_num:, :]
        A_pred = tf.einsum('ij,kj->ik', H, H)
        R_pred = tf.einsum('ij,kj->ik', H_d_out, H_t_out)

        if self.loss_mode == 'RDT' and self.kwargs['use_D'] and self.kwargs['use_T']:
            loss = self.calc_loss(A_pred, self.A)
        else:
            loss = self.calc_loss(R_pred, self.R)
        self.add_loss(loss)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d_out, axis=0),
                tf.expand_dims(H_t_out, axis=0)]

    def get_config(self):
        config = {
            'levels': self.levels,
            'units': self.units,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'loss_mode': self.loss_mode,
            'seed': self.seed
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        H = tf.expand_dims(self.H, axis=0)
        Model.fit(self, x=[H], batch_size=1, epochs=self.epochs, verbose=1,
                  callbacks=[self.es_callback])
        [R_pred, H_d_out, H_t_out] = Model.predict(self, x=[H], batch_size=1, verbose=2)
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
        return [self.R_pred, self.H_d_out, self.H_t_out,None]

    @staticmethod
    def loss_BPR(X_pred, X):
        pos_idxs = tf.where(X == 1)
        pos_rate = tf.reduce_mean(X)
        mask = (tf.random.uniform(X.shape) < (pos_rate * 1.5)).astype('float32')
        neg_idxs = tf.where((1 - X) * mask > 0)
        neg_idxs = tf.random.shuffle(neg_idxs)[:tf.shape(pos_idxs)[0]]
        X_pred_pos = tf.gather_nd(X_pred, pos_idxs)
        X_pred_neg = tf.gather_nd(X_pred, neg_idxs)
        loss = -tf.reduce_mean(tf.math.log_sigmoid(X_pred_pos - X_pred_neg))
        return loss

    def calc_loss(self, X_pred, X):
        loss_LP_pos = tf.reduce_mean(tf.square(
            tf.cast(X == 1, 'float32') * (X_pred - X)))
        loss_LP_neg = tf.reduce_mean(tf.square(
            tf.cast(X == 0, 'float32') * (X_pred - X)))
        loss_LP = loss_LP_pos + 0.2 * loss_LP_neg

        return loss_LP
