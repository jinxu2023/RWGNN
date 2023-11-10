import gc
import random
import numpy as np
import tensorflow as tf

from common_functions.utils import Data


class BatchedModel:

    def __init__(self, batch_config=None, model_config=None):
        self.batch_config = batch_config
        self.model_config = model_config
        tf.compat.v1.set_random_seed(model_config['seed'])
        np.random.seed(model_config['seed'])

        model_name = self.model_config['method']
        exec('import models.' + model_name)
        self.Model = eval('models.' + model_name + '.' + model_name + '_Model')

    def load_data(self, input_data):
        self.D = input_data['D']
        self.T = input_data['T']
        self.D_sim = input_data['D_sim']
        self.T_sim = input_data['T_sim']
        self.R_truth = input_data['R_truth']
        self.R_train = input_data['R_train']
        self.R_pred = np.zeros(self.R_truth.shape, 'float32')
        self.mask = input_data['mask']

        d_num = self.D.shape[0]
        t_num = self.T.shape[0]
        input_feature_dim = self.model_config['input_feature_dim']
        self.H_d = input_data['H_d']
        self.H_t = input_data['H_t']

        # self.H_d = np.ones(Data.d_features.shape, 'float32')
        # self.H_t = np.ones(Data.t_features.shape, 'float32')

        self.H_d_out = None
        self.H_t_out = None
        self.extra_info = None

        if self.batch_config['batch_sampler'] in {'RW', 'NN'}:
            if self.batch_config['sample_DT']:
                self.A = np.r_[np.c_[self.D, self.R_train],
                np.c_[self.R_train.T, self.T]]
            else:
                self.A = np.r_[np.c_[np.zeros((d_num, d_num), 'float32'), Data.R_train],
                np.c_[Data.R_train.T, np.zeros((t_num, t_num), 'float32')]]

    def run_all_batches(self):
        d_num = self.D.shape[0]
        t_num = self.T.shape[0]

        if self.batch_config['batch_sampler'] == 'None':
            self.run_one_batch(list(np.arange(d_num)), list(np.arange(t_num)))
        elif self.batch_config['batch_sampler'] == 'Grid':
            d_batch_size = self.batch_config['d_batch_size']
            t_batch_size = self.batch_config['t_batch_size']
            d_batch_num = d_num // d_batch_size
            t_batch_num = t_num // t_batch_size
            if d_num % d_batch_size > 0:
                d_batch_num += 1
            if t_num % t_batch_size > 0:
                t_batch_num += 1
            d_batch_size = d_num // d_batch_num
            t_batch_size = t_num // t_batch_num

            if self.batch_config['shuffle_dt']:
                d_idxs_all = list(np.random.permutation(d_num))
                t_idxs_all = list(np.random.permutation(t_num))
            else:
                d_idxs_all = list(np.arange(d_num))
                t_idxs_all = list(np.arange(t_num))

            cur_batch_idx = 1
            batch_num = d_batch_num * t_batch_num
            cur_d_idx = 0
            for i in range(d_batch_num - 1):
                sampled_d_idxs = d_idxs_all[cur_d_idx: cur_d_idx + d_batch_size]
                cur_t_idx = 0
                for j in range(t_batch_num - 1):
                    sampled_t_idxs = t_idxs_all[cur_t_idx: cur_t_idx + t_batch_size]
                    print('Batch: ' + str(cur_batch_idx) + '/' + str(batch_num))
                    self.run_one_batch(sampled_d_idxs, sampled_t_idxs)
                    cur_batch_idx += 1
                    cur_t_idx += t_batch_size
                sampled_t_idxs = t_idxs_all[cur_t_idx:]
                print('Batch: ' + str(cur_batch_idx) + '/' + str(batch_num))
                self.run_one_batch(sampled_d_idxs, sampled_t_idxs)
                cur_batch_idx += 1
                cur_d_idx += d_batch_size
            sampled_d_idxs = d_idxs_all[cur_d_idx:]
            cur_t_idx = 0
            for j in range(t_batch_num - 1):
                sampled_t_idxs = t_idxs_all[cur_t_idx:cur_t_idx + t_batch_size]
                print('Batch: ' + str(cur_batch_idx) + '/' + str(batch_num))
                self.run_one_batch(sampled_d_idxs, sampled_t_idxs)
                cur_batch_idx += 1
                cur_t_idx += t_batch_size
            sampled_t_idxs = t_idxs_all[cur_t_idx:]
            print('Batch: ' + str(cur_batch_idx) + '/' + str(batch_num))
            self.run_one_batch(sampled_d_idxs, sampled_t_idxs)
            cur_batch_idx += 1

        elif self.batch_config['batch_sampler'] in {'RW', 'NN'}:
            cur_batch_idx = 1
            batch_num = self.batch_config['batch_num']
            for i in range(batch_num):
                [sampled_d_idxs, sampled_t_idxs] = eval(
                    'self.' + self.batch_config['batch_sampler'] + '_sample()')
                print('Batch: ' + str(cur_batch_idx) + '/' + str(batch_num))
                self.run_one_batch(sampled_d_idxs, sampled_t_idxs)
                cur_batch_idx += 1

    def RW_sample(self):
        path_len = self.batch_config['path_len']
        path_num = self.batch_config['path_num']
        min_node_num = self.batch_config['min_node_num']
        max_node_num = self.batch_config['max_node_num']
        d_num = self.D.shape[0]
        t_num = self.T.shape[0]

        A_cor = [np.flatnonzero(self.A[i, :]) for i in range(d_num + t_num)]

        while 1:
            sampled_d_idxs = set()
            sampled_t_idxs = set()
            start_node = random.randint(0, self.A.shape[0] - 1)

            if len(A_cor[start_node]) == 0:
                continue
            else:
                if start_node < d_num:
                    sampled_d_idxs.add(start_node)
                else:
                    sampled_t_idxs.add(start_node - d_num)
            for i in range(path_num):
                current_node = start_node
                for j in range(path_len):
                    permed_nodes = np.random.permutation(A_cor[current_node])
                    current_node = permed_nodes[0]
                    if current_node < d_num:
                        sampled_d_idxs.add(current_node)
                    else:
                        sampled_t_idxs.add(current_node - d_num)
            if min_node_num <= len(sampled_d_idxs) <= max_node_num \
                    and min_node_num <= len(sampled_t_idxs) <= max_node_num:
                break

        return [list(sampled_d_idxs), list(sampled_t_idxs)]

    def NN_sample(self):
        d_num = self.D.shape[0]

        while 1:
            min_node_num = self.batch_config['min_node_num']
            max_node_num = self.batch_config['max_node_num']
            neigh_level = self.batch_config['neigh_level']
            A_power = np.linalg.matrix_power(self.A + np.eye(self.A.shape[0]), neigh_level)
            A_power[A_power > 0] = 1

            center_node = random.randint(0, self.A.shape[0] - 1)
            sampled_dt_idxs = np.flatnonzero(A_power[center_node, :])
            sampled_d_idxs = []
            sampled_t_idxs = []
            for node in sampled_dt_idxs:
                if node < d_num:
                    sampled_d_idxs.append(node)
                else:
                    sampled_t_idxs.append(node - d_num)
            if min_node_num <= len(sampled_d_idxs) <= max_node_num \
                    and min_node_num <= len(sampled_t_idxs) <= max_node_num:
                break

        return [list(sampled_d_idxs), list(sampled_t_idxs)]

    def run_one_batch(self, d_idxs, t_idxs):
        d_num = len(d_idxs)
        t_num = len(t_idxs)
        _D = self.D[d_idxs, :][:, d_idxs]
        _T = self.T[t_idxs, :][:, t_idxs]
        _D_sim = self.D_sim[d_idxs, :][:, d_idxs]
        _T_sim = self.T_sim[t_idxs, :][:, t_idxs]
        _R_train = self.R_train[d_idxs, :][:, t_idxs]
        _R_truth = self.R_truth[d_idxs, :][:, t_idxs]
        _mask = self.mask[d_idxs, :][:, t_idxs]

        if self.H_d is not None:
            _H_d = self.H_d[d_idxs, :]
        else:
            _H_d = np.eye(d_num + t_num).astype('float32')[:d_num, :]
        if self.H_t is not None:
            _H_t = self.H_t[t_idxs, :]
        else:
            _H_t = np.eye(d_num + t_num).astype('float32')[d_num:, :]

        input_data = [_D, _T, _R_train, _R_truth, _H_d, _H_t, _mask, _D_sim, _T_sim,
                      np.asarray(d_idxs), np.asarray(t_idxs)]
        model = self.Model(input_data=input_data,
                           **self.model_config)

        model.fit(epochs=self.model_config['epochs'])
        [_R_pred, _H_d_out, _H_t_out, extra_info] = model.predict()

        if not np.mean(_R_pred) == 0:
            update_d_idxs = np.repeat(d_idxs, t_num).tolist()
            update_t_idxs = t_idxs * d_num
            self.R_pred[update_d_idxs, update_t_idxs] = (_R_pred / np.max(_R_pred)).reshape((-1,))
        if self.batch_config['batch_sampler'] == 'None':
            self.H_d_out = _H_d_out
            self.H_t_out = _H_t_out
            self.extra_info = extra_info
        self.complete_model_config = model.get_config()

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    def predict(self):
        return [np.asarray(self.R_pred),
                self.H_d_out,
                self.H_t_out,
                self.extra_info]
