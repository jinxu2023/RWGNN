import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import scipy.sparse.linalg


class CLML_Model:
    def __init__(self, input_data=None,
                 theta=2, initial_eta=10 ** -5, tol=5, hyper_paras=(1, 1), epochs=100, **kwargs):
        self.theta = theta
        self.initial_eta = initial_eta
        self.tol = tol
        self.hyper_paras = hyper_paras
        self.epochs = epochs
        self.kwargs = kwargs

        self.init_data(input_data)

        self.eta_d = initial_eta
        self.eta_t = initial_eta

    def init_data(self, input_data):
        D, T, R_train = input_data[:3]
        self.R = tf.convert_to_tensor(R_train, dtype='float32')
        self.D = tf.convert_to_tensor(D, dtype='float32')
        self.T = tf.convert_to_tensor(T, dtype='float32')
        self.D_ind = tf.convert_to_tensor(D > 0, dtype='float32')
        self.T_ind = tf.convert_to_tensor(T > 0, dtype='float32')

        self.R_pred = None
        self.D_pred = tf.random.uniform(D.shape)
        self.T_pred = tf.random.uniform(T.shape)

    def get_config(self):
        config = {
            'theta': self.theta,
            'initial_eta': self.initial_eta,
            'epochs': self.epochs
            # 'hyper_paras': self.hyper_paras,
        }
        return dict(config, **self.kwargs)

    def P_d(self, D_old, eta_d):
        gamma = self.hyper_paras[1]
        X = D_old - eta_d * self.grad_d(D_old)
        U, Sigma, V = sp.linalg.svds(X)
        Sigma = tf.math.maximum(0, Sigma - gamma / eta_d)
        # Sigma = tf.cast(tf.linalg.diag(Sigma), 'float32')
        # D_new = tf.matmul(tf.matmul(U, Sigma), V)
        Sigma = tf.cast(tf.expand_dims(Sigma, axis=0), dtype='float32')
        D_new = tf.matmul(U * Sigma, V)
        return D_new

    def Q_t(self, T_new, T_old, eta_t):
        gamma = self.hyper_paras[1]
        r1 = self.f_t(T_old) + tf.linalg.trace(
            tf.matmul(tf.transpose(T_new - T_old), self.grad_t(T_old)))
        r2 = 1 / (2 * eta_t) * tf.reduce_sum(tf.square(T_new - T_old))
        r3 = gamma * tf.reduce_sum(sp.linalg.svds(T_new, return_singular_vectors=False))
        return r1 + r2 + r3

    def P_t(self, T_old, eta_t):
        gamma = self.hyper_paras[1]
        X = T_old - eta_t * self.grad_t(T_old)
        U, Sigma, V = sp.linalg.svds(X)
        Sigma = tf.math.maximum(0, Sigma - gamma / eta_t)
        # Sigma = tf.cast(tf.linalg.diag(Sigma), 'float32')
        # T_new = tf.matmul(tf.matmul(U, Sigma), V)
        Sigma = tf.cast(tf.expand_dims(Sigma, axis=0), dtype='float32')
        T_new = tf.matmul(U * Sigma, V)
        return T_new

    def Q_d(self, D_new, D_old, eta_d):
        gamma = self.hyper_paras[1]
        r1 = self.f_d(D_old) + tf.linalg.trace(
            tf.matmul(tf.transpose(D_new - D_old), self.grad_d(D_old)))
        r2 = 1 / (2 * eta_d) * tf.reduce_sum(tf.square(D_new - D_old))
        r3 = gamma * tf.reduce_sum(sp.linalg.svds(D_new, return_singular_vectors=False))
        return r1 + r2 + r3

    def grad_d(self, D_pred_new):
        alpha = self.hyper_paras[0]
        r1 = -tf.matmul(tf.matmul(self.R, self.T_pred), tf.transpose(self.R))
        r2 = tf.matmul(tf.matmul(D_pred_new, self.R), tf.transpose(self.R))
        r3 = alpha * self.D_ind * (D_pred_new - self.D_pred)
        return r1 + r2 + r3

    def grad_t(self, T_pred_new):
        beta = self.hyper_paras[0]
        r1 = tf.matmul(tf.matmul(tf.transpose(self.R), self.R), T_pred_new)
        r2 = -tf.matmul(tf.matmul(tf.transpose(self.R), self.D_pred), self.R)
        r3 = beta * self.T_ind * (T_pred_new - self.T_pred)
        return r1 + r2 + r3

    def get_best_d(self):
        n = 1
        initial_eta_d = self.eta_d
        D_pred_new = self.D_pred
        while True:
            try:
                D_pred_new = self.P_d(self.D_pred, self.eta_d)
                if self.L_d(D_pred_new) < self.Q_d(D_pred_new, self.D_pred, self.eta_d):
                    print('best D found after ' + str(n) + ' iters')
                    return D_pred_new
                else:
                    if n == 10:
                        print('best D not found after 10 iters')
                        return D_pred_new
                    self.eta_d = self.eta_d * self.theta
                    n = n + 1
            except:
                print('exception happens after ' + str(n) + ' iters')
                return D_pred_new

    def get_best_t(self):
        n = 1
        initial_eta_t = self.eta_t
        T_pred_new = self.T_pred
        while True:
            try:
                T_pred_new = self.P_t(self.T_pred, self.eta_t)
                if self.L_t(T_pred_new) < self.Q_t(T_pred_new, self.T_pred, self.eta_t):
                    print('best T found after ' + str(n) + ' iters')
                    return T_pred_new
                else:
                    if n == 10:
                        print('best T not found after 10 iters')
                        return T_pred_new
                    self.eta_t = self.eta_t * self.theta
                    n = n + 1
            except:
                print('exception happens after ' + str(n) + ' iters')
                return T_pred_new

    def f_d(self, D_pred):
        alpha = self.hyper_paras[0]
        r1 = tf.reduce_sum(tf.square(tf.matmul(self.R, self.T_pred) - tf.matmul(D_pred, self.R)))
        r2 = tf.reduce_sum(tf.square(self.D_ind * (self.D_pred - self.D)))
        return r1 + alpha * r2

    def L_d(self, D_pred):
        gamma = self.hyper_paras[1]
        return self.f_d(D_pred) + gamma * tf.reduce_sum(sp.linalg.svds(D_pred)[0])

    def f_t(self, T_pred):
        beta = self.hyper_paras[0]
        r1 = tf.reduce_sum(tf.square(tf.matmul(self.R, T_pred) - tf.matmul(self.D_pred, self.R)))
        r2 = tf.reduce_sum(tf.square(self.T_ind * (self.T_pred - self.T)))
        return r1 + beta * r2

    def L_t(self, T_pred):
        gamma = self.hyper_paras[1]
        return self.f_t(T_pred) + gamma * tf.reduce_sum(sp.linalg.svds(T_pred)[0])

    def fit(self, **kwargs):

        best_loss = 10 ** 10
        no_improve_iters = 0
        for i in range(self.epochs):
            print('Epoch ' + str(i + 1) + '/' + str(self.epochs))

            self.D_pred = self.get_best_d()
            self.T_pred = self.get_best_t()

            loss = self.calculate_loss()
            print('loss:' + str(loss.numpy()))

            if loss < best_loss:
                best_loss = loss
                no_improve_iters = 0
            else:
                no_improve_iters += 1
                if no_improve_iters == self.tol:
                    break

            self.R_pred = (tf.matmul(self.R, self.T_pred) + tf.matmul(self.D_pred, self.R)) / 2

    def calculate_loss(self):
        alpha, gamma = self.hyper_paras[:2]
        beta = alpha
        return tf.reduce_sum(
            tf.square(tf.matmul(self.R, self.T_pred) - tf.matmul(self.D_pred, self.R))) \
            + alpha * tf.reduce_sum(tf.square(self.D_ind * (self.D_pred - self.D))) \
            + beta * tf.reduce_sum(tf.square(self.T_ind * (self.T_pred - self.T))) \
            + gamma * (tf.reduce_sum(sp.linalg.svds(self.D_pred)[0]) + tf.reduce_sum(
                sp.linalg.svds(self.T_pred)[0]))

    def predict(self, **kwargs):
        return [self.R_pred, None, None,None]
