import os.path
import tensorflow as tf
import numpy as np


class Configs:
    dataset = None
    method = None
    data_config = None
    extra_data_config = None
    batch_config = None
    model_config = None
    metric_config = None

    data_config_str = None
    full_dataset = None
    batch_config_str = None  # result folder1
    model_config_str = None  # result folder2
    metric_config_str = None  # result file
    result_dir = None
    partial_result_dir = None
    result_file_prefix = None
    partial_result_file_prefix = None

    cur_fold_idx = 0
    cur_tuning_idx = 0

    @classmethod
    def init_configs(cls, dataset=None, method=None, data_config=None, batch_config=None,
                     model_config=None, metric_config=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except BaseException as e:
                print(e)
        tf.compat.v1.set_random_seed(model_config['seed'])
        np.random.seed(model_config['seed'])
        tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=True)
        # tf.debugging.enable_check_numerics()

        cls.dataset = dataset
        cls.method = method
        cls.data_config = data_config
        cls.batch_config = batch_config
        cls.model_config = model_config
        cls.metric_config = metric_config

        cls.data_config_str = cls.get_data_config_str(data_config)
        cls.full_dataset = cls.dataset + '(' + cls.data_config_str + ')'
        cls.batch_config_str = cls.get_batch_config_str(batch_config)
        cls.model_config_str = cls.get_model_config_str(model_config)
        cls.metric_config_str = cls.get_metric_config_str(metric_config)

        cls.result_dir = '..\\..\\3_results\\' + cls.dataset + '\\' + cls.data_config_str + '\\' + \
                         cls.batch_config_str + '\\' + method + '\\' + cls.model_config_str
        cls.partial_result_dir = cls.result_dir + '\\partial_result'
        cls.result_dir = '\\\?\\' + os.path.abspath(cls.result_dir)
        cls.partial_result_dir = '\\\?\\' + os.path.abspath(cls.partial_result_dir)
        cls.result_file_prefix = cls.method + '(' + cls.metric_config_str + ')'
        cls.partial_result_file_prefix = cls.method

    @staticmethod
    def get_data_config_str(data_config):
        s = ''
        if data_config['n_folds'] == 1:
            s += 'val=' + str(data_config['val_prop']) + ',' + 'test=' + str(
                data_config['test_prop']) + ','
        else:
            s += 'n_folds=' + str(data_config['n_folds']) + ','
        s += 'bal=' + str(int(data_config['balance_01'])) + ','

        if s.endswith(','):
            s = s[:-1]
        return s

    @staticmethod
    def get_batch_config_str(batch_config):
        s = 'bs=' + batch_config['batch_sampler'] + ','
        if batch_config['batch_sampler'] == 'Grid':
            s += 's=' + str(batch_config['d_batch_size']) + 'x' + str(
                batch_config['t_batch_size']) + ','
            s += 'ms=' + str(batch_config['min_d_num']) + 'x' + str(batch_config['min_t_num']) + ','
            s += 'sh=' + str(int(batch_config['shuffle_dt'])) + ','
        elif batch_config['batch_sampler'] == 'RW':
            s += 'bn=' + str(batch_config['batch_num']) + ','
            s += 'sdt' + str(int(batch_config['sample_DT'])) + ','
            s += 'pl=' + str(batch_config['path_len']) + ','
            s += 'pn=' + str(batch_config['path_num']) + ','
            s += 'nn=' + str(batch_config['min_node_num']) + \
                 '-' + str(batch_config['max_node_num']) + ','
        elif batch_config['batch_sampler'] == 'NN':
            s += 'bn=' + str(batch_config['batch_num']) + ','
            s += 'sdt=' + str(int(batch_config['sample_DT'])) + ','
            s += 'nl=' + str(batch_config['neigh_level']) + ','
            s += 'nn=' + str(batch_config['min_node_num']) + \
                 '-' + str(batch_config['max_node_num']) + ','

        if s.endswith(','):
            s = s[:-1]
        return s

    @staticmethod
    def get_model_config_str(model_config):
        s = ''

        if model_config['use_d_features']:
            s += '+df,'
        if model_config['use_t_features']:
            s += '+tf,'
        if not model_config['use_D']:
            s += '-D,'
        if not model_config['use_T']:
            s += '-T,'

        s += 'e=' + str(model_config['epochs']) + ','
        s += 'l=' + str(model_config['levels']) + ','
        s += 'u=' + str(model_config['units']) + ','
        s += 'lm=' + str(model_config['loss_mode']) + ','
        s += 'opt=' + str(model_config['optimizer']) + ','
        s += 'idim=' + str(model_config['input_feature_dim']) + ','
        s += 'seed=' + str(model_config['seed']) + ','

        if model_config['method'].startswith('CGRN'):
            s += 'GL=' + model_config['GL_mode'] + ','
            s += 'MI=' + model_config['loss_MI_mode']
            if model_config['loss_MI_mode'] != 'None':
                s += '(' + str(model_config['MI_alg'])
                if model_config['MI_alg'] % 4 != 0:
                    s += ',' + model_config['MI_neg_sampler']
                s += ')'
            s += ','
            s += 'drop=' + model_config['drop_mode']
            if model_config['drop_mode'] != 'None':
                s += '(' + '{0:.2f}'.format(model_config['drop_rate']) + ')'
            s += ','
            s += 'top_k=' + str(model_config['top_k']) + ','
            s += 'GC=' + model_config['GC_mode'] + '(' + model_config['GC_layer_alg'] + '),'
            s += 'l_glo=' + model_config['loss_global_mode'] + ','
            s += 'LP=' + model_config['loss_LP_mode'] + '(' + model_config['loss_LP_alg'] + '),'

        if model_config['method'] == 'RWGNN':
            s += 'h=' + str(model_config['hops']) + ','
            s += 'mins=' + str(model_config['min_step']) + ','
            s += 'maxs=' + str(model_config['max_step']) + ','
            s += 'loss_alg=' + str(model_config['loss_alg']) + ','
            s += 'uP=' + str(int(model_config['use_rw_profile'])) + ','
            if model_config['profile_config'] != 7:
                s += 'pconf=' + str(int(model_config['profile_config'])) + ','
            s += 'uG=' + str(int(model_config['use_GNN'])) + ','

        if model_config['method'] == 'RWGNN2':
            s += 'loss_alg=' + str(model_config['loss_alg']) + ','
            s += 'uP=' + str(int(model_config['use_rw_profile'])) + ','
            s += 'uG=' + str(int(model_config['use_GNN'])) + ','

        if s.endswith(','):
            s = s[:-1]
        return s

    @staticmethod
    def get_metric_config_str(metric_config):
        s = 'mc=' + str(metric_config['metric_group_idx']) + '(' + str(metric_config['cv_metric'])
        if metric_config['best_for_all_metrics']:
            s += '+'
        s += ')'
        if metric_config['best_per_epoch']:
            s += ',pat=' + str(metric_config['patience'])
        return s

    @staticmethod
    def get_hyper_paras_str(hyper_paras):
        if hyper_paras is None:
            return ''

        s = 'hyper_paras='
        for i, para in enumerate(hyper_paras):
            s += '{0:.2e}'.format(para)
            if i < len(hyper_paras) - 1:
                s += ','
        return s
