import os.path

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform

from common_functions.metrics import MetricUtils
from common_functions.configs import Configs
from common_functions.rw_profile import *


class RWGNN_Model(Model):

    def __init__(self, input_data=None,
                 levels=2, units=200, epochs=200, hops=4, min_step=2, max_step=7,
                 optimizer='adam', seed=0, GC_mode='RDT', loss_mode='R', loss_alg='MSE', use_rw_profile=True,
                 profile_config=7, use_GNN=True, hyper_paras=(1, 1), **kwargs):
        super(RWGNN_Model, self).__init__()
        self.levels = levels
        self.units = units
        self.hops = hops
        self.min_step = min_step
        self.max_step = max_step
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.GC_mode = GC_mode
        self.loss_mode = loss_mode
        self.loss_alg = loss_alg
        self.use_rw_profile = use_rw_profile
        self.profile_config = profile_config
        self.use_GNN = use_GNN
        if hyper_paras is not None:
            self.w_D, self.w_T = hyper_paras
        else:
            self.w_D = self.w_T = 0
        self.kwargs = kwargs

        self.init_data(input_data)
        self.GNN_layers = [GNN_Layer(units=units,
                                     activation=tf.nn.sigmoid,
                                     GC_mode=GC_mode,
                                     use_rw_profile=use_rw_profile,
                                     profile_config=profile_config,
                                     seed=seed)
                           for i in range(levels)]

        self.MLP_dt = Dense(units=1,
                            activation=tf.nn.sigmoid,
                            use_bias=False,
                            kernel_initializer=GlorotUniform(self.seed))
        if self.loss_mode == 'RDT':
            self.MLP_dd = Dense(units=1,
                                activation=tf.nn.sigmoid,
                                use_bias=False,
                                kernel_initializer=GlorotUniform(self.seed))
            self.MLP_tt = Dense(units=1,
                                activation=tf.nn.sigmoid,
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
        [D, T, R_train, R_truth, H_d, H_t, mask] = input_data[:7]
        self.d_idxs, self.t_idxs = input_data[-2:]
        self.pos_link_profiles = self.neg_link_profiles = None
        if self.use_rw_profile and self.profile_config != 0:
            self.get_rw_profile()
        self.R_train = tf.convert_to_tensor(R_train, dtype='float32')
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.D = tf.convert_to_tensor(D, dtype='float32')
        self.T = tf.convert_to_tensor(T, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')
        self.H_d = tf.convert_to_tensor(H_d, dtype='float32')
        self.H_t = tf.convert_to_tensor(H_t, dtype='float32')

        self.d_num = D.shape[0]
        self.t_num = T.shape[0]
        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

        self.pos_link_idxs = tf.where((self.R_truth == 1) & (self.mask > 0))
        self.neg_link_idxs = tf.where((self.R_truth == 0) & (self.mask > 0))
        self.pos_link_idxs_train = tf.where((self.R_train == 1) & (self.mask == 1))
        self.neg_link_idxs_train = tf.where((self.R_train == 0) & (self.mask == 1))
        if self.pos_link_idxs_train.shape[0] > self.neg_link_idxs_train.shape[0]:
            self.pos_link_idxs_train = self.pos_link_idxs_train[:self.neg_link_idxs_train.shape[0]]
        elif self.pos_link_idxs_train.shape[0] < self.neg_link_idxs_train.shape[0]:
            self.neg_link_idxs_train = self.neg_link_idxs_train[:self.pos_link_idxs_train.shape[0]]

        if self.loss_mode == 'RDT':
            self.pos_idxs_D, self.neg_idxs_D = self.pos_neg_sampling(self.D)
            self.pos_idxs_T, self.neg_idxs_T = self.pos_neg_sampling(self.T)

    def get_rw_profile(self):
        R_train = Data.R_truth + 0
        mask = np.zeros(Data.mask.shape)
        R_train[mask == Configs.cur_fold_idx + 1] = 0
        mask[Data.mask == Configs.cur_fold_idx + 1] = 3
        mask[(Data.mask > 0) & (Data.mask != Configs.cur_fold_idx + 1)] = 1

        pos_link_profiles_g, neg_link_profiles_g, pos_link_idxs_g, neg_link_idxs_g = get_profiles(
            Data.D, Data.T, R_train, mask, self.hops, self.min_step, self.max_step, Configs.full_dataset)

        re_idxs = []
        for i in range(pos_link_idxs_g.shape[0]):
            d_idx_g, t_idx_g = pos_link_idxs_g[i, :]
            idx_d = np.where(self.d_idxs == d_idx_g)[0]
            idx_t = np.where(self.t_idxs == t_idx_g)[0]
            if len(idx_d) > 0 and len(idx_t) > 0:
                re_idxs += [i]
        pos_link_profiles = pos_link_profiles_g[re_idxs, :]

        re_idxs = []
        for i in range(neg_link_idxs_g.shape[0]):
            d_idx_g, t_idx_g = neg_link_idxs_g[i, :]
            idx_d = np.where(self.d_idxs == d_idx_g)[0]
            idx_t = np.where(self.t_idxs == t_idx_g)[0]
            if len(idx_d) > 0 and len(idx_t) > 0:
                re_idxs += [i]
        neg_link_profiles = neg_link_profiles_g[re_idxs, :]

        profile_flags = [None] * 3
        t = self.profile_config
        for i in range(2, -1, -1):
            profile_flags[i] = t % 2
            t = t // 2
        re = []
        if profile_flags[0]:
            re += [0, 1, 2, 3, 4, 5]
        if profile_flags[1]:
            re += [6, 7, 8, 9]
        if profile_flags[2]:
            re += [10, 11, 12, 13, 14, 15, 16, 17]
        re_num = len(re)
        path_len_num = int(self.max_step - self.min_step + 1)
        prob_num = int(pos_link_profiles.shape[1] / path_len_num)
        pos_link_profiles = pos_link_profiles.reshape((-1, path_len_num, prob_num))
        neg_link_profiles = neg_link_profiles.reshape((-1, path_len_num, prob_num))
        pos_link_profiles = pos_link_profiles[:, :, re].reshape((-1, path_len_num * re_num))
        neg_link_profiles = neg_link_profiles[:, :, re].reshape((-1, path_len_num * re_num))

        self.pos_link_profiles = tf.convert_to_tensor(pos_link_profiles, 'float32')
        self.neg_link_profiles = tf.convert_to_tensor(neg_link_profiles, 'float32')

    def call(self, inputs, training=None, mask=None):
        H_d, H_t = inputs[0][0], inputs[1][0]
        for i in range(self.levels):
            if self.use_rw_profile and self.profile_config != 0:
                H_d, H_t = self.GNN_layers[i]([self.R_train, self.D, self.T, H_d, H_t,
                                               self.pos_link_profiles, self.neg_link_profiles,
                                               self.pos_link_idxs, self.neg_link_idxs])
            else:
                H_d, H_t = self.GNN_layers[i]([self.R_train, self.D, self.T, H_d, H_t,
                                               self.pos_link_idxs, self.neg_link_idxs])

        loss_R = loss_D = loss_T = 0
        if self.loss_mode == 'RDT':
            R_pred = self.predict_score(H_d, H_t, self.pos_link_idxs, self.neg_link_idxs, self.MLP_dt,
                                        self.pos_link_profiles, self.neg_link_profiles)
            D_pred = self.predict_score(H_d, H_d, self.pos_idxs_D, self.neg_idxs_D, self.MLP_dd)
            T_pred = self.predict_score(H_t, H_t, self.pos_idxs_T, self.neg_idxs_T, self.MLP_tt)
            if self.loss_alg == 'MSE':
                loss_R = self.loss_MSE(R_pred, self.R_train)
                loss_D = self.loss_MSE(D_pred, self.D)
                loss_T = self.loss_MSE(T_pred, self.T)
            elif self.loss_alg == 'BPR':
                loss_R = self.loss_BPR(R_pred, self.pos_link_idxs_train, self.neg_link_idxs_train)
                loss_D = self.loss_BPR(D_pred, self.pos_idxs_D, self.neg_idxs_D)
                loss_T = self.loss_BPR(T_pred, self.pos_idxs_T, self.neg_idxs_T)
        elif self.loss_mode == 'R':
            R_pred = self.predict_score(H_d, H_t, self.pos_link_idxs, self.neg_link_idxs, self.MLP_dt,
                                        self.pos_link_profiles, self.neg_link_profiles)
            if self.loss_alg == 'MSE':
                loss_R = self.loss_MSE(R_pred, self.R_train)
            elif self.loss_alg == 'BPR':
                loss_R = self.loss_BPR(R_pred, self.pos_link_idxs_train, self.neg_link_idxs_train)
        loss = loss_R + self.w_D * loss_D + self.w_T * loss_T
        self.add_loss(loss)

        if Configs.metric_config['metric_group_idx'] == 0:
            metric_name = Configs.metric_config['cv_metric']
            if len(metric_name.split('_')[0][3:]) == 0:
                top_n = 10 ** 8
            else:
                top_n = int(metric_name.split('_')[0][3:])
            cv_flag = metric_name.split('_')[1]
            if cv_flag == 'train':
                flag = 1
            elif cv_flag == 'val':
                flag = 2
            elif cv_flag == 'test':
                flag = 3
            auc = MetricUtils.calc_auc(R_pred, self.R_truth, self.mask, (top_n,), (flag,))
            self.add_metric(auc, name=Configs.metric_config['cv_metric'], aggregation='mean')

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d, axis=0),
                tf.expand_dims(H_t, axis=0)]

    def predict_score(self, H_r, H_c, pos_link_idxs, neg_link_idxs, MLP,
                      pos_link_profiles=None, neg_link_profiles=None):
        scores = tf.zeros((H_r.shape[0], H_c.shape[0]))
        if self.use_GNN:
            H_r_pos = tf.gather(H_r, pos_link_idxs[:, 0], axis=0)
            H_c_pos = tf.gather(H_c, pos_link_idxs[:, 1], axis=0)
            H_r_neg = tf.gather(H_r, neg_link_idxs[:, 0], axis=0)
            H_c_neg = tf.gather(H_c, neg_link_idxs[:, 1], axis=0)
            if pos_link_profiles is None or neg_link_profiles is None:
                H_pos = tf.concat([H_r_pos, H_c_pos], axis=1)
                H_neg = tf.concat([H_r_neg, H_c_neg], axis=1)
            else:
                H_pos = tf.concat([H_r_pos, H_c_pos, pos_link_profiles], axis=1)
                H_neg = tf.concat([H_r_neg, H_c_neg, neg_link_profiles], axis=1)
        else:
            H_pos = pos_link_profiles
            H_neg = neg_link_profiles
        scores_pos = MLP(H_pos).reshape((-1,))
        scores_neg = MLP(H_neg).reshape((-1,))
        scores = tf.tensor_scatter_nd_update(scores, pos_link_idxs, scores_pos)
        scores = tf.tensor_scatter_nd_update(scores, neg_link_idxs, scores_neg)
        return scores

    def get_config(self):
        config = {
            'levels': self.levels,
            'units': self.units,
            'epochs': self.epochs,
            'hops': self.hops,
            'min_step': self.min_step,
            'max_step': self.max_step,
            'optimizer': self._optimizer,
            'seed': self.seed,
            'loss_mode': self.loss_mode,
            'loss_alg': self.loss_alg,
            'use_rw_profile': self.use_rw_profile,
            'profile_config': self.profile_config,
            'use_GNN': self.use_GNN
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
        extra_info = {
            'MLP_weights': self.MLP_dt.weights[0].numpy()
        }
        return [self.R_pred, self.H_d_out, self.H_t_out, extra_info]

    @staticmethod
    def loss_MSE(X_pred, X):
        loss_pos = tf.reduce_mean(tf.square(
            tf.cast(X == 1, 'float32') * (X_pred - X)))
        loss_neg = tf.reduce_mean(tf.square(
            tf.cast(X == 0, 'float32') * (X_pred - X)))
        loss = loss_pos + 0.2 * loss_neg
        return loss

    @staticmethod
    def pos_neg_sampling_by_row(X):
        pos_idxs = tf.where(X == 1).astype('int32')
        pos_num_per_row = tf.reduce_sum((X == 1).astype('int32'), axis=1)
        neg_idxs = tf.zeros((0, 2), 'int32')
        for i in tf.range(tf.shape(X)[0]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(neg_idxs, tf.TensorShape([None, 2]))])
            zero_idxs = tf.where(X[i, :] == 0).astype('int32')
            zero_idxs = tf.random.shuffle(zero_idxs)
            pos_num = pos_num_per_row[i]
            zero_idxs = zero_idxs[:pos_num]
            _neg_idxs = tf.concat(
                [tf.ones((pos_num, 1), 'int32') * i, zero_idxs.reshape((-1, 1))], axis=1)
            neg_idxs = tf.concat([neg_idxs, _neg_idxs], axis=0)
        return pos_idxs, neg_idxs

    @staticmethod
    def pos_neg_sampling(X):
        pos_idxs = tf.where(X == 1)
        if tf.shape(pos_idxs)[0] > 100:
            r = 1.5
        else:
            r = 10.
        pos_rate = tf.reduce_mean(X)
        mask = (tf.random.uniform(X.shape) < (pos_rate * r)).astype('float32')
        neg_idxs = tf.where((1 - X) * mask > 0)
        neg_idxs = tf.random.shuffle(neg_idxs)[:tf.shape(pos_idxs)[0]]
        return pos_idxs, neg_idxs

    @staticmethod
    def loss_BPR(X_pred, pos_idxs, neg_idxs):
        X_pred_pos = tf.gather_nd(X_pred, pos_idxs)
        X_pred_neg = tf.gather_nd(X_pred, neg_idxs)
        loss = -tf.reduce_mean(tf.math.log_sigmoid(X_pred_pos - X_pred_neg))
        return loss


class GNN_Layer(Layer):
    def __init__(self, units=200, GC_mode='RDT', activation=tf.nn.relu, use_rw_profile=True,
                 profile_config=7, seed=0):
        super(GNN_Layer, self).__init__()
        self.units = units
        self.GC_mode = GC_mode
        self.activation = activation
        self.use_rw_profile = use_rw_profile
        self.profile_config = profile_config
        self.seed = seed

    def build(self, input_shapes):
        self.dense_d = Dense(units=self.units,
                             activation=self.activation,
                             use_bias=False,
                             kernel_initializer=GlorotUniform(self.seed))
        self.dense_t = Dense(units=self.units,
                             activation=self.activation,
                             use_bias=False,
                             kernel_initializer=GlorotUniform(self.seed + 1))
        self.dense_pro = Dense(units=1,
                               activation=tf.nn.sigmoid,
                               use_bias=False,
                               kernel_initializer=GlorotUniform(self.seed + 2))
        self.W_d = self.add_weight(name='W_d',
                                   shape=(input_shapes[3][-1], self.units),
                                   initializer=GlorotUniform(self.seed))
        self.W_t = self.add_weight(name='W_t',
                                   shape=(input_shapes[4][-1], self.units),
                                   initializer=GlorotUniform(self.seed))

        self.dense_d2 = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed + 2))
        self.dense_t2 = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed + 3))
        self.dense_dd = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))
        self.dense_dt = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))
        self.dense_td = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))
        self.dense_tt = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))

    def call(self, inputs, **kwargs):
        if self.use_rw_profile and self.profile_config != 0:
            R, D, T, H_d, H_t, pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs = inputs
        else:
            R, D, T, H_d, H_t, pos_link_idxs, neg_link_idxs = inputs
        H_d_concat_list, H_t_concat_list = [H_d], [H_t]

        n_neigh_dt = tf.reduce_sum(R, axis=1, keepdims=True)
        n_neigh_td = tf.reduce_sum(R, axis=0, keepdims=True)
        n_neigh_dt = n_neigh_dt + (n_neigh_dt == 0).astype('float32')
        n_neigh_td = n_neigh_td + (n_neigh_td == 0).astype('float32')
        n_neigh_dt_norm = n_neigh_dt ** -0.5
        n_neigh_td_norm = n_neigh_td ** -0.5
        R_norm = n_neigh_dt_norm * R * n_neigh_td_norm

        W_link = tf.ones(R_norm.shape)
        if self.use_rw_profile and self.profile_config != 0:
            pos_link_weights = self.dense_pro(pos_link_profiles).reshape((-1,))
            neg_link_weights = self.dense_pro(neg_link_profiles).reshape((-1,))
            W_link = tf.zeros(R.shape)
            W_link = tf.tensor_scatter_nd_update(W_link, pos_link_idxs, pos_link_weights)
            W_link = tf.tensor_scatter_nd_update(W_link, neg_link_idxs, neg_link_weights)

        H_dt = self.activation((W_link * R_norm) @ H_t)
        H_td = self.activation((W_link.T * R_norm.T) @ H_d)
        H_d_concat_list += [H_dt]
        H_t_concat_list += [H_td]

        if self.GC_mode == 'RDT':
            n_neigh_dd = tf.reduce_sum(D, axis=1, keepdims=True)
            n_neigh_tt = tf.reduce_sum(T, axis=1, keepdims=True)
            n_neigh_dd = n_neigh_dd + (n_neigh_dd == 0).astype('float32')
            n_neigh_tt = n_neigh_tt + (n_neigh_tt == 0).astype('float32')
            n_neigh_dd_norm = n_neigh_dd ** -0.5
            n_neigh_tt_norm = n_neigh_tt ** -0.5
            D_norm = n_neigh_dd_norm * D * n_neigh_dd_norm.T
            T_norm = n_neigh_tt_norm * T * n_neigh_tt_norm.T

            H_dd = self.activation(D_norm @ H_d)
            H_tt = self.activation(T_norm @ H_t)
            H_d_concat_list += [H_dd]
            H_t_concat_list += [H_tt]

        H_d_out = self.dense_d(tf.concat(H_d_concat_list, axis=1))
        H_t_out = self.dense_t(tf.concat(H_t_concat_list, axis=1))
        return H_d_out, H_t_out
