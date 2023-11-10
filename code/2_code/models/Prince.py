import numpy as np
import tensorflow as tf
import scipy.spatial.distance as dist


class Prince_Model:
    def __init__(self, input_data=None, hyper_paras=(1, 1), **kwargs):
        self.alpha, self.c = hyper_paras
        self.kwargs = kwargs
        self.init_data(input_data)

    def init_data(self, input_data):
        self.R_train, self.S_d, self.S_t = input_data[2], input_data[7], input_data[8]

        self.Y = np.zeros(self.R_train.shape, 'float32')
        for i in range(self.S_t.shape[0]):
            d_idxs = tf.squeeze(tf.where(self.R_train[:, i] == 1))
            S_d_sel = tf.gather(self.S_d, d_idxs)
            self.Y[:, i] = tf.reduce_max(S_d_sel, axis=0)
        self.Y = 1 / (1 + np.exp(self.c * self.Y + np.log(9999))).astype('float32')

    def get_config(self):
        return self.kwargs

    def predict(self, **kwargs):
        return [self.R_pred, None, None, None]

    def fit(self, **kwargs):
        inv_mat = np.linalg.pinv(tf.eye(self.S_t.shape[0]) - self.alpha * self.S_t)
        self.R_pred = (1 - self.alpha) * tf.matmul(self.Y, inv_mat)
