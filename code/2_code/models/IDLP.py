import numpy as np
import tensorflow as tf


class IDLP_Model:
    def __init__(self, input_data=None, tol=5, hyper_paras=(1, 1), epochs=200, **kwargs):
        super(IDLP_Model, self).__init__()
        self.tol = tol
        self.hyper_paras = hyper_paras
        self.epochs = epochs
        self.kwargs = kwargs
        self.init_data(input_data)

    def init_data(self, input_data):
        D, T, R_train = input_data[:3]
        self.R = tf.convert_to_tensor(R_train, dtype='float32')
        self.R_pred = None

        D = tf.convert_to_tensor(D, dtype='float32')
        D_row_sum = tf.reduce_sum(D, axis=1, keepdims=True)
        D_row_norm = D_row_sum ** -0.5
        nan_idxs = tf.where(tf.math.is_nan(D_row_norm))
        D_row_norm = tf.tensor_scatter_nd_update(D_row_norm, nan_idxs,
                                                 tf.zeros(nan_idxs.shape[0]))
        inf_idxs = tf.where(tf.math.is_inf(D_row_norm))
        D_row_norm = tf.tensor_scatter_nd_update(D_row_norm, inf_idxs,
                                                 tf.zeros(inf_idxs.shape[0]))
        D_col_norm = tf.transpose(D_row_norm)
        self.S_d = D_row_norm * D * D_col_norm

        T = tf.convert_to_tensor(T, dtype='float32')
        T_row_sum = tf.reduce_sum(T, axis=1, keepdims=True)
        T_row_norm = T_row_sum ** -0.5
        nan_idxs = tf.where(tf.math.is_nan(T_row_norm))
        T_row_norm = tf.tensor_scatter_nd_update(T_row_norm, nan_idxs,
                                                 tf.zeros(nan_idxs.shape[0]))
        inf_idxs = tf.where(tf.math.is_inf(T_row_norm))
        T_row_norm = tf.tensor_scatter_nd_update(T_row_norm, inf_idxs,
                                                 tf.zeros(inf_idxs.shape[0]))
        T_col_norm = tf.transpose(T_row_norm)
        self.S_t = T_row_norm * T * T_col_norm

        self.R_pred = tf.random.uniform(self.R.shape)
        self.S_d_pred = tf.random.uniform(self.S_d.shape)
        self.S_t_pred = tf.random.uniform(self.S_t.shape)

    def get_config(self):
        config = {
            'epochs': self.epochs
            # 'hyper_paras': self.hyper_paras,
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        alpha, gamma = self.hyper_paras[:4]
        alpha_dot, gamma_dot = alpha, gamma
        beta, beta_dot = 1 - alpha, 1 - alpha_dot

        best_loss = 10 ** 10
        no_improve_iters = 0
        for i in range(self.epochs):
            print('Epoch ' + str(i + 1) + '/' + str(self.epochs))
            try:
                self.S_d_pred = self.S_d + gamma * tf.einsum('ij,kj->ik', self.R_pred,
                                                             self.R_pred)
                self.R_pred = beta * tf.matmul(
                    tf.linalg.inv(tf.eye(self.S_d.shape[0]) - alpha * self.S_d_pred),
                    self.R)
                self.S_t_pred = self.S_t + gamma_dot * tf.einsum('ji,jk->ik', self.R_pred,
                                                                 self.R_pred)
                self.R_pred = beta_dot * tf.matmul(self.R,
                                                   tf.linalg.inv(
                                                       tf.eye(self.S_t.shape[
                                                                  0]) - alpha_dot * self.S_t_pred))
            except BaseException as e:
                print(e)

            loss = self.calculate_loss()
            print('loss:' + str(loss.numpy()))

            if loss < best_loss:
                best_loss = loss
                no_improve_iters = 0
            else:
                no_improve_iters += 1
                if no_improve_iters == self.tol:
                    break

    def calculate_loss(self):
        alpha, gamma = self.hyper_paras[:2]
        alpha_dot, gamma_dot = alpha, gamma
        beta, beta_dot = 1 - alpha, 1 - alpha_dot
        mu = (1 / alpha) - 1
        nu = 1 / (2 * gamma)
        zeta = (1 / alpha_dot) - 1
        eta = 1 / (2 * gamma_dot)

        loss1 = tf.linalg.trace(tf.einsum('ji,jk,kl->il', self.R_pred,
                                          tf.linalg.eye(self.R.shape[0]) - self.S_d,
                                          self.R_pred)) \
                + mu * tf.reduce_sum((self.R_pred - self.R) ** 2) \
                + nu * tf.reduce_sum((self.S_d_pred - self.S_d) ** 2)

        loss2 = tf.linalg.trace(tf.einsum('ij,jk,lk->il', self.R_pred,
                                          tf.linalg.eye(self.R.shape[1]) - self.S_t,
                                          self.R_pred)) \
                + zeta * tf.reduce_sum((self.R_pred - self.R) ** 2) \
                + eta * tf.reduce_sum((self.S_t_pred - self.S_t) ** 2)

        return loss1 + loss2

    def predict(self, **kwargs):
        return [np.asarray(self.R_pred), None, None,None]
