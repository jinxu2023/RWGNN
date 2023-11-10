import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import scipy.sparse.linalg


class RWRH_Model:
    def __init__(self, input_data=None, hyper_paras=(1, 1, 1), **kwargs):
        self.hyper_paras = hyper_paras
        self.kwargs = kwargs

        self.init_data(input_data)

    def init_data(self, input_data):
        gamma, lambd, eta = self.hyper_paras[:3]
        self.D_sim, self.T_sim, self.R = input_data[7], input_data[8], input_data[2]

        R_row_sums = np.sum(self.R, axis=1, keepdims=True)
        R_row_sums[R_row_sums == 0] = -1
        R_col_sums = np.sum(self.R, axis=0, keepdims=True)
        R_col_sums[R_col_sums == 0] = 1
        M_TD = lambd * self.R / R_col_sums
        M_DT = lambd * (self.R / R_row_sums).T

        mask_D = R_row_sums == -1
        D_row_sums = np.sum(self.D_sim, axis=1, keepdims=True)
        D_row_sums[D_row_sums == 0] = -1
        M_D = mask_D * self.D_sim / D_row_sums + (1 - mask_D) * (1 - lambd) * self.D_sim / D_row_sums

        mask_T = (R_col_sums == -1).T
        T_row_sums = np.sum(self.T_sim, axis=1, keepdims=True)
        T_row_sums[T_row_sums == 0] = -1
        M_T = mask_T * self.T_sim / T_row_sums + (1 - mask_T) * (1 - lambd) * self.T_sim / T_row_sums

        self.M = np.r_[np.c_[M_D, M_TD], np.c_[M_DT, M_T]]

        A1 = np.r_[self.D_sim, self.R.T]
        A1_row_sums = np.sum(A1, axis=1, keepdims=True)
        A1_row_sums[A1_row_sums == 0] = -1
        A1 = A1 / A1_row_sums

        A2 = np.r_[self.R, self.T_sim]
        A2_row_sums = np.sum(A1, axis=1, keepdims=True)
        A2_row_sums[A2_row_sums == 0] = -1
        A2 = A2 / A2_row_sums

        self.A = np.c_[eta * A1, (1 - eta) * A2]
        self.A[np.isnan(self.A)] = 0

    def get_config(self):
        return self.kwargs

    def fit(self, **kwargs):
        gamma = self.hyper_paras[0]
        pred = (1 - gamma) * self.A @ np.linalg.inv(np.eye(self.M.shape[0]) - gamma * self.M)
        self.R_pred = pred[:self.R.shape[0], self.R.shape[0]:]

    def predict(self, **kwargs):
        return [self.R_pred, None, None, None]
