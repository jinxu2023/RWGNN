import os
import pickle
import time
import numpy as np
import tensorflow as tf

from common_functions.utils import Data, Utils
from common_functions.configs import Configs
from common_functions.metrics import MetricUtils
from common_functions.batch import BatchedModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 0

dataset = 'Drugbank&HPRD'
method = 'RWGNN'
batch_sampler = 'None'
use_D = True
use_T = True
levels = 2
units = 64
optimizer = 'rmsprop'
loss_mode = 'R'
epochs = 200
metric_group_idx = 3
cv_metric = 'auc_train'
best_per_epoch = False

load_saved_results = True

data_config = {
    'n_folds': 10,
    'val_prop': 0.0,
    'test_prop': 0.1,
    'balance_01': True,
}
batch_config = {
    'batch_sampler': batch_sampler,  # None, Grid, RW, NN
    'shuffle_dt': True,  # batch_sampler==Grid
    'd_batch_size': 2000,  # batch_sampler==Grid
    't_batch_size': 2000,  # batch_sampler==Grid
    'min_d_num': 1000,  # batch_sampler==Grid
    'min_t_num': 1000,  # batch_sampler==Grid
    'batch_num': 100,  # batch_sampler==RW or NN
    'sample_DT': True,  # batch_sampler==RW or NN
    'path_len': 100,  # batch_sampler==RW
    'path_num': 200,  # batch_sampler==RW
    'min_node_num': 500,  # batch_sampler==RW or NN
    'max_node_num': 2500,
    'neigh_level': 3,  # batch_sampler==NN
}
model_config = {
    'method': method,
    'use_d_features': False,
    'use_t_features': False,
    'use_D': use_D,
    'use_T': use_T,
    'epochs': epochs,
    'levels': levels,
    'units': units,
    'loss_mode': loss_mode,
    'optimizer': optimizer,
    'input_feature_dim': 200,
    'seed': seed
}

if method.startswith('RWGNN'):
    RWGNN_config = {
        'min_step': 2,
        'max_step': 7,
        'use_rw_profile': True,
        'profile_config': 7,
        'use_GNN': True,
        'loss_alg': 'MSE',
        'hops': 4
    }
    model_config = dict(model_config, **RWGNN_config)

metric_names = MetricUtils.metric_groups[metric_group_idx]
metric_config = {
    'metric_group_idx': metric_group_idx,
    'metric_names': metric_names,
    'best_for_all_metrics': False,
    'cv_metric': cv_metric,
    'cv_metric_idx': metric_names.index(cv_metric),
    'best_per_epoch': best_per_epoch,
    'patience': 500
}

Configs.init_configs(dataset, method, data_config, batch_config, model_config, metric_config)
data = Data.load_data(dataset, data_config, model_config)
Utils.make_dir(Configs.result_dir, Configs.partial_result_dir)

if method == 'STARGCN':
    paras_groups = (
        tuple(5.0 ** np.arange(-4, 5)),
    )
elif method == 'IDLP':
    paras_groups = (
        tuple(5.0 ** np.arange(-4, 5)),
        tuple(5.0 ** np.arange(-4, 5))
    )
elif method == 'CLML':
    paras_groups = (
        tuple(10.0 ** np.arange(-4, 5)),
        tuple(10.0 ** np.arange(-4, 5))
    )
elif method == 'Prince':
    paras_groups = (
        (0.01, 0.1, 0.2, 0.5),
        (-10, -15, -20, -25, -30)
    )
elif method == 'RWRH':
    paras_groups = (
        (0.8, 0.85, 0.9, 0.95),
        (0.8, 0.85, 0.9, 0.95),
        (0.8, 0.85, 0.9, 0.95)
    )
elif method == 'RWGNN':
    if loss_mode == 'RDT':
        paras_groups = (
            tuple(5.0 ** np.arange(-4, 3)),
            tuple(5.0 ** np.arange(-4, 3))
        )
    else:
        paras_groups = ()
else:
    paras_groups = ()

n_paras = len(paras_groups)
n_folds = data_config['n_folds']
n_tunings = 1
tuning_size = ()
for paras in paras_groups:
    n_tunings *= len(paras)
    tuning_size += (len(paras),)

# build result and partial result
result = {
    'dataset': dataset,
    'method': method,
    'data_config': data_config,
    'batch_config': batch_config,
    'model_config': None,
    'metric_config': metric_config,

    'paras_groups': paras_groups,
    'tuning_size': tuning_size,
    'n_tunings': n_tunings,
    'n_folds': n_folds,
    'all_metrics': -np.ones((n_folds, n_tunings, len(metric_names))),

    'best_paras_per_fold': np.zeros((n_folds, n_paras)),
    'best_paras': [],
    'best_cv_metric': -1,
    'best_metrics': -np.ones((len(metric_names),)),
    'best_metrics_per_fold': -np.ones((n_folds, len(metric_names))),
    'best_R_pred': None,
    'best_R_pred_per_fold': [None] * data_config['n_folds'],
    'best_d_features': None,
    'best_t_features': None,
    'best_d_features_per_fold': [None] * data_config['n_folds'],
    'best_t_features_per_fold': [None] * data_config['n_folds'],
    'extra_info': None,

    'last_fold_idx': 0,
    'last_tuning_idx': 0,
    'finish': False
}

partial_result = {
    'dataset': dataset,
    'method': method,
    'data_config': data_config,
    'model_config': None,
    'cur_fold_idx': None,
    'cur_tuning_idx': None,
    'hyper_paras': None,

    'd_features': None,
    't_features': None,
    'extra_info': None,
    'R_pred': None,
    'D_pred': None,
    'T_pred': None,
}

(_result, old_timestamp) = Utils.check_finish(Configs.result_dir, Configs.result_file_prefix)
if old_timestamp is not None:
    result = _result
    last_fold_idx = result['last_fold_idx']
    last_tuning_idx = result['last_tuning_idx']
else:
    last_fold_idx = -1
    last_tuning_idx = -1

old_path = None
for cur_fold_idx in range(0, n_folds):
    for cur_tuning_idx in range(0, n_tunings):
        if cur_fold_idx * n_tunings + cur_tuning_idx <= last_fold_idx * n_tunings + last_tuning_idx:
            continue

        Configs.cur_fold_idx, Configs.cur_tuning_idx = cur_fold_idx, cur_tuning_idx
        print('Current Fold:', cur_fold_idx + 1, '/', n_folds)
        print('Parameter Tuning:', cur_tuning_idx + 1, '/', n_tunings)
        result['last_fold_idx'] = cur_fold_idx
        result['last_tuning_idx'] = cur_tuning_idx
        if cur_fold_idx == n_folds - 1 and cur_tuning_idx == n_tunings - 1:
            result['finish'] = True
        if n_paras > 0:
            cur_paras_idxs = np.unravel_index(cur_tuning_idx, tuning_size)
            cur_paras = [paras_groups[i][j] for (i, j) in enumerate(cur_paras_idxs)]
        else:
            cur_paras_idxs = None
            cur_paras = None
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        # check partial result
        old_partial_results = None
        if load_saved_results is True:
            old_partial_results = Utils.check_partial_finish(
                Configs.partial_result_dir, Configs.partial_result_file_prefix, cur_fold_idx, cur_paras)

        # fill partial result
        update = False
        if old_partial_results is None:
            update = True
            model_config['method'] = method
            model_config['hyper_paras'] = cur_paras
            batched_model = BatchedModel(batch_config, model_config)

            input_data = {
                'R_truth': data.R_truth,
                'D': data.D,
                'T': data.T,
                'D_sim': data.D_sim,
                'T_sim': data.T_sim,
                'H_d': data.d_features,
                'H_t': data.t_features
            }
            if data_config['n_folds'] == 1:
                input_data['R_train'] = data.R_train
                input_data['mask'] = data.mask
            else:
                R_train = data.R_truth + 0
                mask = np.zeros(data.mask.shape)
                R_train[(R_train == 1) & (data.mask == cur_fold_idx + 1)] = 0
                mask[data.mask == cur_fold_idx + 1] = 3
                mask[(data.mask > 0) & (data.mask != cur_fold_idx + 1)] = 1
                input_data['R_train'] = R_train
                input_data['mask'] = mask

            batched_model.load_data(input_data)
            batched_model.run_all_batches()

            partial_result['model_config'] = batched_model.complete_model_config
            if result['model_config'] is None:
                result['model_config'] = batched_model.complete_model_config

            [R_pred, d_features, t_features, extra_info] = batched_model.predict()
            partial_result['R_pred'] = np.squeeze(R_pred)
            partial_result['extra_info'] = extra_info
            if d_features is not None and t_features is not None:
                partial_result['d_features'] = np.squeeze(d_features)
                partial_result['t_features'] = np.squeeze(t_features)

            metrics = np.asarray(
                MetricUtils.calc_metrics(R_pred, data.R_truth, input_data['mask'], metric_group_idx))
            metrics_dict = dict(zip(metric_names, metrics))
            partial_result['metrics_dict' + str(metric_config['metric_group_idx'])] = metrics_dict
            partial_result['cur_paras'] = cur_paras
            partial_result['cur_fold_idx'] = cur_fold_idx
            partial_result['cur_tuning_idx'] = cur_tuning_idx

        else:
            partial_result = old_partial_results
            if not partial_result.keys().__contains__('extra_info'):
                partial_result['extra_info'] = None
            if result['model_config'] is None:
                result['model_config'] = partial_result['model_config']

            if ('metrics_dict' + str(metric_config['metric_group_idx']) in partial_result and
                    len(partial_result['metrics_dict' + str(metric_config['metric_group_idx'])])
                    == len(metric_names)):
                R_pred = partial_result['R_pred']
                metrics_dict = partial_result['metrics_dict' + str(metric_config['metric_group_idx'])]
                metrics = list(metrics_dict.values())
            else:
                update = True
                if partial_result['R_pred'] is None and partial_result['d_features'] is not None:
                    R_pred = tf.einsum('ij,kj->ik',
                                       partial_result['d_features'],
                                       partial_result['t_features']).numpy()
                else:
                    R_pred = partial_result['R_pred']

                if data_config['n_folds'] == 1:
                    mask = data.mask + 0
                else:
                    mask = np.zeros(data.mask.shape)
                    mask[data.mask == cur_fold_idx + 1] = 3
                    mask[(data.mask > 0) & (data.mask != cur_fold_idx + 1)] = 1

                metrics = np.asarray(
                    MetricUtils.calc_metrics(R_pred, data.R_truth, mask, metric_group_idx))
                metrics_dict = dict(zip(metric_names, metrics))
                partial_result['metrics_dict' + str(metric_group_idx)] = metrics_dict

        s = ''
        for metric_name, metric in metrics_dict.items():
            s += metric_name + ': ' + '{0:.4f}'.format(metric) + ', '
        print(s)
        Utils.format_result(result, partial_result, metrics, cur_fold_idx, cur_tuning_idx, cur_paras)

        # save partial result
        if load_saved_results and update:
            save_path = Configs.partial_result_dir + '\\fold=' + str(cur_fold_idx + 1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            paras_str = Configs.get_hyper_paras_str(cur_paras)
            if paras_str != '':
                save_file = Configs.partial_result_file_prefix + '(' + paras_str + ').pkl'
            else:
                save_file = Configs.partial_result_file_prefix + '.pkl'
            with open(save_path + '\\' + save_file, 'wb') as file:
                s = pickle.dumps(partial_result)
                file.write(s)

        # save global result
        if result['finish']:
            Utils.format_final_result(result)

        cur_timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        temp_path = Configs.result_dir + '\\~$^' + Configs.result_file_prefix \
                    + '@' + str(cur_timestamp) + '.pkl'
        save_path = Configs.result_dir + '\\^' + Configs.result_file_prefix \
                    + '@' + str(cur_timestamp) + '.pkl'
        if old_path is None:
            old_path = Configs.result_dir + '\\^' + Configs.result_file_prefix \
                       + '@' + str(old_timestamp) + '.pkl'

        with open(temp_path, 'wb') as file:
            s = pickle.dumps(result)
            file.write(s)
            if os.path.isfile(old_path):
                os.remove(old_path)
            file.close()
            if result['finish']:
                save_path = save_path.replace('\\^', '\\')
            os.rename(temp_path, save_path)

        old_path = save_path
        if result['finish']:
            Utils.display_result(Configs.result_dir, Configs.result_file_prefix, display_mat=True)
            save_path2 = save_path.replace('.pkl', '.txt')
            with open(save_path2, 'w') as file:
                result_str = Utils.result_to_str(result)
                file.write('----------result summary----------\n\n' + result_str)
                file.close()
