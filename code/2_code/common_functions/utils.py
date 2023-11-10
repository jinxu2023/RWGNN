import glob
import os
import pickle
import time

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans

from common_functions.configs import Configs


class Data:
    initial_mask = None
    R_truth = None
    D = None
    T = None
    D_sim = None
    T_sim = None
    R_train = None
    d_features = None
    t_features = None
    mask = None
    d_names = None
    d_ids = None
    t_names = None
    t_ids = None
    d_levels = None
    t_levels = None
    d_depth = None

    dataset = None
    full_dataset = None
    data_config = None
    model_config = None

    @classmethod
    def load_data(cls, dataset, data_config=None, model_config=None):
        cls.full_dataset = cls.dataset = dataset
        cls.data_config = data_config
        cls.model_config = model_config
        if data_config is not None:
            cls.full_dataset += '(' + Configs.get_data_config_str(data_config) + ')'

        processed_data_path = "..\\..\\1_processed_data\\" + cls.full_dataset + ".mat"
        if not os.path.isfile(processed_data_path):
            cls.generate_processed_data()

        data = scio.loadmat(processed_data_path)
        cls.R_truth = data['R'].astype('float32')
        cls.D_sim = cls.D = data['D'].astype('float32')
        cls.T_sim = cls.T = data['T'].astype('float32')
        if data.__contains__('D_sim'):
            cls.D_sim = data['D_sim'].astype('float32')
        if data.__contains__('T_sim'):
            cls.T_sim = data['T_sim'].astype('float32')
        cls.mask = data['mask'].astype('float32')

        if data.__contains__('R_train'):
            cls.R_train = data['R_train'].astype('float32')
        if data.__contains__('d_names'):
            cls.d_names = np.squeeze(data['d_names'])
            for i, d_name in enumerate(cls.d_names):
                cls.d_names[i] = d_name[0]
        if data.__contains__('d_ids'):
            cls.d_ids = np.squeeze(data['d_ids'])
        if data.__contains__('t_names'):
            cls.t_names = np.squeeze(data['t_names'])
            for i, t_name in enumerate(cls.t_names):
                cls.t_names[i] = t_name[0]
        if data.__contains__('t_ids'):
            cls.t_ids = np.squeeze(data['t_ids'])
        if data.__contains__('d_levels'):
            cls.d_levels = np.squeeze(data['d_levels'])

        if cls.model_config is None:
            return Data
        if cls.model_config['use_d_features']:
            cls.d_features = Utils.pca(data['d_features'], model_config['input_feature_dim'])
        else:
            if model_config['input_feature_dim'] is not None:
                cls.d_features = Utils.one_hot(cls.R_truth.shape[0], model_config['input_feature_dim'])
            else:
                one_hot = np.eye(cls.R_truth.shape[0] + cls.R_truth.shape[1]).astype('float32')
                cls.d_features = one_hot[:cls.R_truth.shape[0], :]
        if cls.model_config['use_t_features']:
            cls.t_features = Utils.pca(data['t_features'], model_config['input_feature_dim'])
        else:
            if model_config['input_feature_dim'] is not None:
                cls.t_features = Utils.one_hot(cls.R_truth.shape[1], model_config['input_feature_dim'])
            else:
                one_hot = np.eye(cls.R_truth.shape[0] + cls.R_truth.shape[1]).astype('float32')
                cls.t_features = one_hot[cls.R_truth.shape[0]:, :]
        if not cls.model_config['use_D']:
            cls.D = cls.D_sim = np.zeros(data['D'].shape, 'float32')
        if not cls.model_config['use_T']:
            cls.T = cls.T_sim = np.zeros(data['T'].shape, 'float32')
        return Data

    @classmethod
    def generate_processed_data(cls):
        path = "..\\..\\0_original_data\\" + cls.dataset + ".mat"
        data = scio.loadmat(path)
        R_source = None
        D_source = None
        T_source = None
        D_sim_source = None
        T_sim_source = None
        d_ids_source = None
        d_names_source = None
        t_ids_source = None
        t_names_source = None
        d_features_source = None
        t_features_source = None
        d_levels_source = None
        if cls.dataset == 'Luos':
            R_source = 'Drug_Target'
            D_source = 'Drug_Drug'
            T_source = 'Target_Target'
            D_sim_source = 'Drug_Drug_sim'
            T_sim_source = 'Target_Target_sim'
            d_ids_source = 'Drug_ids'
            d_names_source = 'Drug_names'
            t_ids_source = 'Target_ids'
            t_names_source = 'Target_names'
            d_features_source = 'Drug_SideEffect'
            t_features_source = 'Target_Disease'
        elif cls.dataset == 'Drugbank&HPRD':
            R_source = 'drug_target'
            D_source = 'drug_drug'
            T_source = 'target_target'
            d_ids_source = 'drug_id'
            d_names_source = 'drug_name'
            t_ids_source = 'target_id'
            t_names_source = 'target_name'
        elif cls.dataset == 'Zhengs':
            R_source = 'R'
            D_source = 'D'
            T_source = 'T'
            D_sim_source = 'D_sim'
            T_sim_source = 'T_sim'
            d_names_source = 'Dname'
            t_names_source = 'Tname'

        cls.R_truth = data[R_source].astype('float32')
        cls.D = data[D_source].astype('float32')
        cls.T = data[T_source].astype('float32')

        if data.__contains__('initial_mask'):
            cls.initial_mask = data['initial_mask'].astype('float32')
        if data.__contains__(D_sim_source):
            cls.D_sim = data[D_sim_source].astype('float32')
        if data.__contains__(T_sim_source):
            cls.T_sim = data[T_sim_source].astype('float32')
        if data.__contains__(d_features_source):
            cls.d_features = data[d_features_source].astype('float32')
        if data.__contains__(t_features_source):
            cls.t_features = data[t_features_source].astype('float32')
        if data.__contains__(d_names_source):
            cls.d_names = np.squeeze(data[d_names_source])
            for i, d_name in enumerate(cls.d_names):
                cls.d_names[i] = d_name[0]
        if data.__contains__(t_names_source):
            cls.t_names = np.squeeze(data[t_names_source])
            for i, t_name in enumerate(cls.t_names):
                cls.t_names[i] = t_name[0]
        if data.__contains__(d_ids_source):
            cls.d_ids = np.squeeze(data[d_ids_source])
        if data.__contains__(t_ids_source):
            cls.t_names = np.squeeze(data[t_ids_source])
        if data.__contains__(d_levels_source):
            cls.d_levels = np.squeeze(data[d_levels_source])
        cls.split_train_val_test()

        save_path = "..\\..\\1_processed_data\\" + cls.full_dataset + ".mat"
        save_data = {
            'R': cls.R_truth,
            'D': cls.D,
            'T': cls.T,
            'mask': cls.mask,
        }
        if cls.R_train is not None:
            save_data['R_train'] = np.asarray(cls.R_train)
        if cls.D_sim is not None:
            save_data['D_sim'] = np.asarray(cls.D_sim)
        if cls.T_sim is not None:
            save_data['T_sim'] = np.asarray(cls.T_sim)
        if cls.d_features is not None:
            save_data['d_features'] = np.asarray(cls.d_features)
        if cls.t_features is not None:
            save_data['t_features'] = np.asarray(cls.t_features)
        if cls.d_names is not None:
            save_data['d_names'] = cls.d_names
        if cls.d_ids is not None:
            save_data['d_ids'] = cls.d_ids
        if cls.t_names is not None:
            save_data['t_names'] = cls.t_names
        if cls.t_ids is not None:
            save_data['t_ids'] = cls.t_ids
        if cls.d_levels is not None:
            save_data['d_levels'] = cls.d_levels

        scio.savemat(save_path, save_data, do_compression=True)

    @classmethod
    def filter_RDT(cls):
        cls.R_truth[cls.R_truth < cls.data_config['R_threshold']] = 0
        cls.D[cls.D < cls.data_config['D_threshold']] = 0
        cls.T[cls.T < cls.data_config['T_threshold']] = 0

        if cls.data_config['binary_R']:
            cls.R_truth = (cls.R_truth > 0).astype('float32')
        else:
            cls.R_truth = cls.R_truth / np.max(cls.R_truth)
        if cls.data_config['binary_D']:
            cls.D = (cls.D > 0).astype('float32')
        else:
            cls.D = cls.D / np.max(cls.D)
        if cls.data_config['binary_T']:
            cls.T = (cls.T > 0).astype('float32')
        else:
            cls.T = cls.T / np.max(cls.T)

    @classmethod
    def split_train_val_test(cls):
        if cls.initial_mask is not None:
            cls.mask = cls.initial_mask
            cls.R_train = cls.R_truth * (cls.mask == 1)
            return

        if cls.data_config['n_folds'] == 1:
            val_prop = cls.data_config['val_prop']
            test_prop = cls.data_config['test_prop']
            balance_01 = cls.data_config['balance_01']

            cls.mask = np.zeros(cls.R_truth.shape)
            (non_zero_idxs_r, non_zero_idxs_c) = np.nonzero(cls.R_truth)
            (zero_idxs_r, zero_idxs_c) = np.nonzero(cls.R_truth == 0)

            n_non_zero_train = int(len(non_zero_idxs_r) * (1 - val_prop - test_prop))
            n_non_zero_val = int(len(non_zero_idxs_r) * val_prop)
            n_non_zero_test = len(non_zero_idxs_r) - n_non_zero_train - n_non_zero_val
            shuffle_flag = np.argsort(np.random.random(non_zero_idxs_r.shape))
            non_zero_idxs_r_shuffled = np.take_along_axis(non_zero_idxs_r, shuffle_flag, axis=0)
            non_zero_idxs_c_shuffled = np.take_along_axis(non_zero_idxs_c, shuffle_flag, axis=0)
            cls.mask[(non_zero_idxs_r_shuffled[:n_non_zero_train],
                      non_zero_idxs_c_shuffled[:n_non_zero_train])] = 1
            cls.mask[(non_zero_idxs_r_shuffled[n_non_zero_train:n_non_zero_train + n_non_zero_val],
                      non_zero_idxs_c_shuffled[
                      n_non_zero_train:n_non_zero_train + n_non_zero_val])] = 2
            cls.mask[(non_zero_idxs_r_shuffled[n_non_zero_train + n_non_zero_val:],
                      non_zero_idxs_c_shuffled[n_non_zero_train + n_non_zero_val:])] = 3

            if balance_01:
                n_zero_train = n_non_zero_train
                n_zero_val = n_non_zero_val
                n_zero_test = n_non_zero_test
            else:
                n_zero_train = int(len(zero_idxs_r) * (1 - val_prop - test_prop))
                n_zero_val = int(len(zero_idxs_r) * val_prop)
                n_zero_test = len(zero_idxs_r) - n_zero_train - n_zero_val

            shuffle_flag = np.argsort(np.random.random(zero_idxs_r.shape))
            zero_idxs_r_shuffled = np.take_along_axis(zero_idxs_r, shuffle_flag, axis=0)
            zero_idxs_c_shuffled = np.take_along_axis(zero_idxs_c, shuffle_flag, axis=0)
            cls.mask[(zero_idxs_r_shuffled[:n_zero_train],
                      zero_idxs_c_shuffled[:n_zero_train])] = 1
            cls.mask[(zero_idxs_r_shuffled[n_zero_train:n_zero_train + n_zero_val],
                      zero_idxs_c_shuffled[n_zero_train:n_zero_train + n_zero_val])] = 2
            cls.mask[(zero_idxs_r_shuffled[n_zero_train + n_zero_val:
                                           n_zero_train + n_zero_val + n_zero_test],
                      zero_idxs_c_shuffled[n_zero_train + n_zero_val:
                                           n_zero_train + n_zero_val + n_zero_test])] = 3
            cls.R_train = cls.R_truth * (cls.mask == 1)
        else:
            n_folds = cls.data_config['n_folds']
            balance_01 = cls.data_config['balance_01']

            cls.mask = np.zeros(cls.R_truth.shape)
            (non_zero_idxs_r, non_zero_idxs_c) = np.nonzero(cls.R_truth)
            (zero_idxs_r, zero_idxs_c) = np.nonzero(cls.R_truth == 0)

            n_non_zeros = len(non_zero_idxs_r)
            n_per_fold = int(n_non_zeros / n_folds) + 1
            fold_flags = np.asarray([i + 1 for i in range(n_folds)] * n_per_fold)[:n_non_zeros]
            shuf_idxs = np.random.permutation(n_non_zeros)
            fold_flags_shuf = fold_flags[shuf_idxs]
            cls.mask[non_zero_idxs_r, non_zero_idxs_c] = fold_flags_shuf

            n_zeros = len(zero_idxs_r)
            if balance_01:
                n_per_fold = int(n_non_zeros / n_folds) + 1
                fold_flags = np.asarray([i + 1 for i in range(n_folds)] * n_per_fold)[:n_non_zeros]
                fold_flags = np.concatenate([fold_flags, np.zeros((n_zeros - n_non_zeros,), 'int32')])
            else:
                n_per_fold = int(n_zeros / n_folds) + 1
                fold_flags = np.asarray([i + 1 for i in range(n_folds)] * n_per_fold)[:n_zeros]
            shuf_idxs = np.random.permutation(n_zeros)
            fold_flags_shuf = fold_flags[shuf_idxs]
            cls.mask[zero_idxs_r, zero_idxs_c] = fold_flags_shuf

    @staticmethod
    def get_R(dataset, data_config):
        if Data.R_truth is not None:
            return Data.R_truth
        else:
            full_dataset = dataset + '(' + Configs.get_data_config_str(data_config) + ')'
            processed_data_path = "..\\..\\1_processed_data\\" + full_dataset + ".mat"
            if os.path.isfile(processed_data_path):
                data = scio.loadmat(processed_data_path)
                return data['R'].astype('float32')


class Utils:
    @staticmethod
    def make_dir(result_dir, partial_result_dir):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(partial_result_dir):
            os.makedirs(partial_result_dir)

    @staticmethod
    def check_finish(result_dir, result_file_prefix):
        path = result_dir + '\\^' + result_file_prefix + '@*.pkl'
        largest_timestamp = 0
        for file_name in glob.glob(path):
            if file_name.startswith('~$'):
                continue
            index = file_name.index('@')
            timestamp = int(file_name[index + 1:index + 15])
            if timestamp > largest_timestamp:
                largest_timestamp = timestamp
        if largest_timestamp > 0:
            path = path.replace('*', str(largest_timestamp))
            with open(path, 'rb') as file:
                result = pickle.loads(file.read())
                if result['finish'] is False:
                    return result, largest_timestamp
        return None, None

    @staticmethod
    def check_partial_finish(partial_result_dir, partial_result_file_prefix, cur_fold_idx, cur_paras):
        paras_str = Configs.get_hyper_paras_str(cur_paras)
        if paras_str != '':
            path = (partial_result_dir + '\\fold=' + str(cur_fold_idx + 1)
                    + '\\' + partial_result_file_prefix + '(' + paras_str + ').pkl')
        else:
            path = partial_result_dir + '\\fold=' + str(cur_fold_idx + 1) + '\\' + partial_result_file_prefix + '.pkl'
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                result = pickle.loads(file.read())
                return result
        return None

    @staticmethod
    def format_result(result, partial_result, metrics, cur_fold_idx, cur_tuning_idx, cur_paras):
        cv_metric = result['metric_config']['cv_metric']
        cv_metric_idx = result['metric_config']['metric_names'].index(cv_metric)
        best_for_all_metrics = result['metric_config']['best_for_all_metrics']

        result['all_metrics'][cur_fold_idx, cur_tuning_idx, :] = metrics
        result['best_metrics'] = np.maximum(result['best_metrics'], metrics)

        cv_metric_val = metrics[cv_metric_idx]
        if cv_metric_val > result['best_cv_metric']:
            result['best_cv_metric'] = cv_metric_val
            result['best_paras'] = cur_paras
            result['best_d_features'] = partial_result['d_features']
            result['best_t_features'] = partial_result['t_features']
            result['best_R_pred'] = partial_result['R_pred']
            result['extra_info'] = partial_result['extra_info']
            if not best_for_all_metrics:
                result['best_metrics'] = metrics

        if cv_metric_val > result['best_metrics_per_fold'][cur_fold_idx, cv_metric_idx]:
            result['best_R_pred_per_fold'][cur_fold_idx] = partial_result['R_pred']
            result['best_d_features_per_fold'][cur_fold_idx] = partial_result['d_features']
            result['best_t_features_per_fold'][cur_fold_idx] = partial_result['t_features']
            if best_for_all_metrics:
                result['best_metrics_per_fold'][cur_fold_idx, :] = np.maximum(
                    result['best_metrics_per_fold'][cur_fold_idx, :], metrics)
            else:
                result['best_metrics_per_fold'][cur_fold_idx, :] = metrics

    @staticmethod
    def format_final_result(result):
        best_for_all_metrics = result['metric_config']['best_for_all_metrics']
        cv_metric = result['metric_config']['cv_metric']
        cv_metric_idx = result['metric_config']['metric_names'].index(cv_metric)

        if best_for_all_metrics:
            result['best_metrics_per_fold'] = np.max(result['best_metrics'], axis=1)
        else:
            for fold_idx in range(result['n_folds']):
                metrics_cur_fold = result['all_metrics'][fold_idx, :]
                best_tuning_idx = np.argmax(metrics_cur_fold[:, cv_metric_idx])
                result['best_metrics_per_fold'][fold_idx, :] = metrics_cur_fold[best_tuning_idx, :]
                best_paras_idxs = np.unravel_index(best_tuning_idx, result['tuning_size'])
                best_paras = [result['paras_groups'][i][j] for (i, j) in enumerate(best_paras_idxs)]
                result['best_paras_per_fold'][fold_idx, :] = best_paras

        result['avg_metrics_all_folds'] = np.mean(result['best_metrics_per_fold'], axis=0)

    @staticmethod
    def get_result(result_dir, result_file_prefix, timestamp=None):
        result_dir = result_dir + '\\' + result_file_prefix + '@*.pkl'
        if timestamp is None:
            files = glob.glob(result_dir)
            if len(files) == 0:
                raise Exception('result not exist')
            index = files[-1].index('@')
            timestamp = files[-1][index + 1:index + 15]
        result_dir = result_dir.replace('*', timestamp)

        with open(result_dir, 'rb') as file:
            result = pickle.loads(file.read())
            return result

    @staticmethod
    def display_result(result_dir, result_file_prefix, timestamp=None, display_mat=False):
        result = Utils.get_result(result_dir, result_file_prefix, timestamp)
        result_str = Utils.result_to_str(result, show_metrics_first=False)
        print('----------result summary----------\n\n' + result_str)
        if not display_mat:
            return result

        plt.figure()
        plt.suptitle('method: ' + result['method'] + '\ndataset: ' + result['dataset'])
        plt.subplot(1, 2, 1)
        if result['best_R_pred'] is not None:
            plt.imshow(result['best_R_pred'], cmap='Greys_r')
        else:
            plt.imshow(np.einsum('ij,kj->ik',
                                 result['best_d_features'],
                                 result['best_t_features']), cmap='Greys_r')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(Data.get_R(result['dataset'], result['data_config']), cmap='Greys_r')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.show()
        return result

    @staticmethod
    def display_overall_result(dataset, methods, data_config, batch_config, model_config,
                               metric_config):
        plt.figure(dpi=1000, figsize=(4, 4))
        plt.subplots_adjust(hspace=0.01, wspace=0.02, left=0.02, right=0.98, bottom=0.02, top=0.98)
        plt.subplot(2, 4, 1)
        plt.title('Ground-Truth', y=0.98, fontdict={'fontsize': 9})
        Data.load_data(dataset, data_config, model_config)
        plt.imshow(Data.R_truth, cmap='Greys_r')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        for i, method in enumerate(methods):
            Configs.init_configs(dataset, method, data_config, batch_config,
                                 dict(model_config, **{'method': method}), metric_config)
            result_path = Configs.result_dir + '\\' + Configs.result_file_prefix + '@*.pkl'

            files = glob.glob(result_path)
            if len(files) == 0:
                raise Exception('result not exist')
            index = files[-1].index('@')
            timestamp = files[-1][index + 1:index + 15]
            result_path = result_path.replace('*', timestamp)

            with open(result_path, 'rb') as file:
                if method == 'CGRN':
                    method = 'SRGL'
                elif method == 'CMRL':
                    method = 'MVGRL'

                result = pickle.loads(file.read())
                plt.subplot(2, 4, i + 2)
                plt.title(method.replace('_', '-'), y=0.98, fontdict={'fontsize': 9})
                if result['best_R_pred'] is not None:
                    plt.imshow(result['best_R_pred'], cmap='Greys_r')
                else:
                    plt.imshow(result['best_d_features'] @ result['best_t_features'].T, cmap='Greys_r')
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])

        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig('..\\..\\3_results\\' + dataset + '\\' + Configs.data_config_str + '\\' +
                    Configs.batch_config_str + '\\overall_results.png')
        # plt.show()

    @staticmethod
    def result_to_str(result, show_metrics_first=True):
        s1 = 'metrics:\n'
        if result['data_config']['n_folds'] == 1:
            metrics = result['best_metrics']
        else:
            metrics = result['avg_metrics_all_folds']
        for i, metric_name in enumerate(result['metric_config']['metric_names']):
            s1 += metric_name + ': ' + '{0:.4f}'.format(metrics[i])
            if i < len(metrics) - 1:
                s1 += ', '
                if i % 3 == 2:
                    s1 += '\n'
            else:
                s1 += '\n\n'

        s1 += 'best_paras: ' + str(result['best_paras']) + '\n'
        s2 = 'dataset: ' + result['dataset'] + '\n'
        s2 += 'method: ' + result['method'] + '\n\n'
        s2 += 'data_config:\n' + str(result['data_config']).replace(',', ',\n') + '\n\n'
        s2 += 'batch_config:\n' + str(result['batch_config']).replace(',', ',\n') + '\n\n'
        s2 += 'model_config:\n' + str(result['model_config']).replace(',', ',\n') + '\n\n'
        s2 += 'metric_config:\n' + str(result['metric_config']).replace(',', ',\n') + '\n\n'
        s2 += 'paras_groups:\n' + str(result['paras_groups']).replace('),',
                                                                      '),\n') + '\n\n'

        all_metrics = result['all_metrics']
        all_metrics = all_metrics.reshape((-1, all_metrics.shape[-1]))
        r_num, c_num = all_metrics.shape
        s2 += 'all_metrics:\n'
        for i in range(r_num):
            for j in range(c_num):
                s2 += '{0:.4f}'.format(all_metrics[i][j])
                if i == r_num - 1 and j == c_num - 1:
                    s2 += '\n\n'
                elif j == c_num - 1:
                    s2 += ',\n'
                else:
                    s2 += ', '

        if show_metrics_first:
            return s1 + s2
        else:
            return s2 + s1

    @staticmethod
    def pca(dataMat, n):
        if n is None:
            return np.asarray(dataMat, 'float32')

        def zeroMean(dataMat):
            meanVal = np.mean(dataMat, axis=0)
            newData = dataMat - meanVal
            return newData, meanVal

        newData, meanVal = zeroMean(dataMat)
        covMat = np.cov(newData, rowvar=0)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        eigValIndice = np.argsort(eigVals)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
        n_eigVect = eigVects[:, n_eigValIndice]
        lowDDataMat = np.matmul(newData, n_eigVect)
        return np.asarray(lowDDataMat, 'float32')

    @staticmethod
    def kmeans(X, c):
        Z = tf.zeros((X.shape[0], c))
        class_labels = KMeans(n_clusters=c).fit_predict(X)
        one_idxs = np.stack((np.arange(0, X.shape[0]), class_labels), axis=1)
        Z = tf.tensor_scatter_nd_update(Z, one_idxs, tf.ones(X.shape[0]))
        return Z

    @staticmethod
    def one_hot(rows, cols):
        if cols is None:
            cols = rows
        if cols >= rows:
            return np.eye(rows, cols)
        else:
            eye = np.eye(cols)
            one_hot_matrix = np.eye(cols)
            for i in range(int(rows / cols)):
                one_hot_matrix = np.r_[one_hot_matrix, eye]
            one_hot_matrix = one_hot_matrix[:rows, :]
            return one_hot_matrix.astype('float32')
