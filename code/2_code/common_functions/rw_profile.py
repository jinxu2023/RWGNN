import tensorflow as tf
import numpy as np
import os
import scipy.io as scio
import scipy.sparse as sp
from common_functions.utils import Configs, Data

profile_len = 18


def matrix_power(X, k):
    t = X
    for i in range(k - 1):
        t = t @ X
    return t


def subgraph_sampling(D, T, R, center_d_idx, center_t_idx, hops):
    D_power = matrix_power(D + tf.eye(D.shape[0]), hops)
    T_power = matrix_power(T + tf.eye(T.shape[0]), hops)
    sampled_d_idxs = tf.where(D_power[center_d_idx, :] >= 1)[:, 0]
    sampled_t_idxs = tf.where(T_power[center_t_idx, :] >= 1)[:, 0]
    D_sub = tf.gather(D, sampled_d_idxs, axis=0)
    D_sub = tf.gather(D_sub, sampled_d_idxs, axis=1)
    T_sub = tf.gather(T, sampled_t_idxs, axis=0)
    T_sub = tf.gather(T_sub, sampled_t_idxs, axis=1)
    R_sub = tf.gather(R, sampled_d_idxs, axis=0)
    R_sub = tf.gather(R_sub, sampled_t_idxs, axis=1)
    A_sub = tf.concat([tf.concat([D_sub, R_sub], axis=1),
                       tf.concat([R_sub.T, T_sub], axis=1)], axis=0)

    return sampled_d_idxs, sampled_t_idxs, D_sub, T_sub, R_sub, A_sub


def cal_profile(D, T, R, A, d_idx, t_idx, min_step, max_step):
    d_num, t_num = D.shape[0], T.shape[0]
    p_ds_D = []
    p_ds_Ap = []
    p_ds_Am = []
    p_ts_T = []
    p_ts_Ap = []
    p_ts_Am = []

    p_dt_Rp = []
    p_dt_Rm = []
    p_dt_Ap = []
    p_dt_Am = []

    p_ads_D = []
    p_ats_T = []
    p_ads_Rp = []
    p_ats_Rp = []
    p_ads_Rm = []
    p_ats_Rm = []
    p_adat_Ap = []
    p_adat_Am = []

    r_sum_D = tf.reduce_sum(D, axis=1, keepdims=True)
    c_sum_D = tf.reduce_sum(D, axis=0, keepdims=True)
    r_sum_D = r_sum_D + (r_sum_D == 0).astype('float32')
    c_sum_D = c_sum_D + (c_sum_D == 0).astype('float32')
    D_norm = (r_sum_D ** -0.5) * D * (c_sum_D ** -0.5)
    D_prob = matrix_power(D_norm, min_step)
    for i in range(max_step - min_step + 1):
        p_ds_D += [D_prob[d_idx, d_idx]]
        p_ads_D += [tf.linalg.trace(D_prob) / d_num]
        D_prob = D_prob @ D_norm
    del D, D_norm, D_prob

    r_sum_T = tf.reduce_sum(T, axis=1, keepdims=True)
    c_sum_T = tf.reduce_sum(T, axis=0, keepdims=True)
    r_sum_T = r_sum_T + (r_sum_T == 0).astype('float32')
    c_sum_T = c_sum_T + (c_sum_T == 0).astype('float32')
    T_norm = (r_sum_T ** -0.5) * T * (c_sum_T ** -0.5)
    T_prob = matrix_power(T_norm, min_step)
    for i in range(max_step - min_step + 1):
        p_ts_T += [T_prob[t_idx, t_idx]]
        p_ats_T += [tf.linalg.trace(T_prob) / t_num]
        T_prob = T_prob @ T_norm
    del T, T_norm, T_prob

    Rp = tf.tensor_scatter_nd_update(R, [[d_idx, t_idx]], [1])
    r_sum_Rp = tf.reduce_sum(Rp, axis=1, keepdims=True)
    c_sum_Rp = tf.reduce_sum(Rp, axis=0, keepdims=True)
    r_sum_Rp = r_sum_Rp + (r_sum_Rp == 0).astype('float32')
    c_sum_Rp = c_sum_Rp + (c_sum_Rp == 0).astype('float32')
    Rp_norm = (r_sum_Rp ** -0.5) * Rp * (c_sum_Rp ** -0.5)
    Rp_norm_s = tf.concat([tf.concat([tf.zeros((d_num, d_num)), Rp_norm], axis=1),
                           tf.concat([Rp_norm.T, tf.zeros((t_num, t_num))], axis=1)], axis=0)
    Rp_prob = matrix_power(Rp_norm_s, min_step)
    for i in range(max_step - min_step + 1):
        p_dt_Rp += [Rp_prob[d_idx, d_num + t_idx]]
        p_ads_Rp += [tf.linalg.trace(Rp_prob[:d_num, :d_num]) / d_num]
        p_ats_Rp += [tf.linalg.trace(Rp_prob[d_num:, d_num:]) / t_num]
        Rp_prob = Rp_prob @ Rp_norm_s
    del Rp, Rp_norm, Rp_norm_s, Rp_prob

    Rm = tf.tensor_scatter_nd_update(R, [[d_idx, t_idx]], [0])
    r_sum_Rm = tf.reduce_sum(Rm, axis=1, keepdims=True)
    c_sum_Rm = tf.reduce_sum(Rm, axis=0, keepdims=True)
    r_sum_Rm = r_sum_Rm + (r_sum_Rm == 0).astype('float32')
    c_sum_Rm = c_sum_Rm + (c_sum_Rm == 0).astype('float32')
    Rm_norm = (r_sum_Rm ** -0.5) * Rm * (c_sum_Rm ** -0.5)
    Rm_norm_s = tf.concat([tf.concat([tf.zeros((d_num, d_num)), Rm_norm], axis=1),
                           tf.concat([Rm_norm.T, tf.zeros((t_num, t_num))], axis=1)], axis=0)
    Rm_prob = matrix_power(Rm_norm_s, min_step)
    for i in range(max_step - min_step + 1):
        p_dt_Rm += [Rm_prob[d_idx, d_num + t_idx]]
        p_ads_Rm += [tf.linalg.trace(Rm_prob[:d_num, :d_num]) / d_num]
        p_ats_Rm += [tf.linalg.trace(Rm_prob[d_num:, d_num:]) / t_num]
        Rm_prob = Rm_prob @ Rm_norm_s
    del Rm, Rm_norm, Rm_norm_s, Rm_prob

    Ap = tf.tensor_scatter_nd_update(A, [[d_idx, d_num + t_idx], [d_num + t_idx, d_idx]], [1, 1])
    r_sum_Ap = tf.reduce_sum(Ap, axis=1, keepdims=True)
    c_sum_Ap = tf.reduce_sum(Ap, axis=0, keepdims=True)
    r_sum_Ap = r_sum_Ap + (r_sum_Ap == 0).astype('float32')
    c_sum_Ap = c_sum_Ap + (c_sum_Ap == 0).astype('float32')
    Ap_norm = (r_sum_Ap ** -0.5) * Ap * (c_sum_Ap ** -0.5)
    Ap_prob = matrix_power(Ap_norm, min_step)
    for i in range(max_step - min_step + 1):
        p_ds_Ap += [Ap_prob[d_idx, d_idx]]
        p_ts_Ap += [Ap_prob[d_num + t_idx, d_num + t_idx]]
        p_dt_Ap += [Ap_prob[d_idx, d_num + t_idx]]
        p_adat_Ap += [tf.reduce_sum(Ap_prob[:d_num, d_num:]) / d_num]
        Ap_prob = Ap_prob @ Ap_norm
    del Ap, Ap_norm, Ap_prob

    Am = tf.tensor_scatter_nd_update(A, [[d_idx, d_num + t_idx], [d_num + t_idx, d_idx]], [0, 0])
    r_sum_Am = tf.reduce_sum(Am, axis=1, keepdims=True)
    c_sum_Am = tf.reduce_sum(Am, axis=0, keepdims=True)
    r_sum_Am = r_sum_Am + (r_sum_Am == 0).astype('float32')
    c_sum_Am = c_sum_Am + (c_sum_Am == 0).astype('float32')
    Am_norm = (r_sum_Am ** -0.5) * Am * (c_sum_Am ** -0.5)
    Am_prob = matrix_power(Am_norm, min_step)
    for i in range(max_step - min_step + 1):
        p_ds_Am += [Am_prob[d_idx, d_idx]]
        p_ts_Am += [Am_prob[d_num + t_idx, d_num + t_idx]]
        p_dt_Am += [Am_prob[d_idx, d_num + t_idx]]
        p_adat_Am += [tf.reduce_sum(Am_prob[:d_num, d_num:]) / d_num]
        Am_prob = Am_prob @ Am_norm
    del Am, Am_norm, Am_prob

    profile = np.c_[p_ds_D, p_ds_Ap, p_ds_Am, p_ts_T, p_ts_Ap, p_ts_Am, p_dt_Rp, p_dt_Rm,
    p_dt_Ap, p_dt_Am, p_ads_D, p_ats_T, p_ads_Rp, p_ats_Rp, p_ads_Rm, p_ats_Rm,
    p_adat_Ap, p_adat_Am].reshape((-1,))

    return profile


def cal_profiles(D, T, R, mask, hops, min_step, max_step):
    d_num, t_num = D.shape[0], T.shape[0]
    pos_d_idxs, pos_t_idxs = np.where((mask > 0) & (R == 1))
    neg_d_idxs, neg_t_idxs = np.where((mask > 0) & (R == 0))
    pos_num, neg_num = len(pos_d_idxs), len(neg_d_idxs)
    pos_link_profiles = np.zeros((pos_num, profile_len * (max_step - min_step + 1)), 'float32')
    neg_link_profiles = np.zeros((neg_num, profile_len * (max_step - min_step + 1)), 'float32')

    for i, d_idx, t_idx in zip(range(pos_num), pos_d_idxs, pos_t_idxs):
        sampled_d_idxs, sampled_t_idxs, D_sub, T_sub, R_sub, A_sub = subgraph_sampling(
            D, T, R, d_idx, t_idx, hops)
        d_idx_sub = np.where(sampled_d_idxs == d_idx)[0][0]
        t_idx_sub = np.where(sampled_t_idxs == t_idx)[0][0]
        profile = cal_profile(D_sub, T_sub, R_sub, A_sub, d_idx_sub, t_idx_sub, min_step, max_step)
        pos_link_profiles[i, :] = profile
        print(i)

    for i, d_idx, t_idx in zip(range(neg_num), neg_d_idxs, neg_t_idxs):
        sampled_d_idxs, sampled_t_idxs, D_sub, T_sub, R_sub, A_sub = subgraph_sampling(
            D, T, R, d_idx, t_idx, hops)
        d_idx_sub = np.where(sampled_d_idxs == d_idx)[0][0]
        t_idx_sub = np.where(sampled_t_idxs == t_idx)[0][0]
        profile = cal_profile(D_sub, T_sub, R_sub, A_sub, d_idx_sub, t_idx_sub, min_step, max_step)
        pos_link_profiles[i, :] = profile
        print(i)

    pos_link_idxs = np.stack([pos_d_idxs, pos_t_idxs], axis=1)
    neg_link_idxs = np.stack([neg_d_idxs, neg_t_idxs], axis=1)

    return pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs


def get_profiles(D, T, R, mask, hops, min_step, max_step, full_dataset):
    profile_dir = '../../1_processed_data/rw_profiles'
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    if Configs.data_config['n_folds'] > 1:
        full_dataset = full_dataset[:-2] + ',fold=' + str(Configs.cur_fold_idx + 1) + ')'
    profile_file = profile_dir + '/' + full_dataset + '_rw_profile(hops=' \
                   + str(hops) + ',min_step=' + str(min_step) \
                   + ',max_step=' + str(max_step) + ').mat'
    if os.path.isfile(profile_file):
        dic = scio.loadmat(profile_file)
        return dic['pos_link_profiles'], dic['neg_link_profiles'], dic['pos_link_idxs'], dic['neg_link_idxs']
    else:
        pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs = cal_profiles(
            D, T, R, mask, hops, min_step, max_step)
        scio.savemat(profile_file, {
            'pos_link_profiles': pos_link_profiles,
            'neg_link_profiles': neg_link_profiles,
            'pos_link_idxs': pos_link_idxs,
            'neg_link_idxs': neg_link_idxs
        }, do_compression=True)
        return pos_link_profiles, neg_link_profiles, pos_link_idxs, neg_link_idxs
