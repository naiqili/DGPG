# -*- coding: utf-8 -*-
# Author: Jack Lee
# GitHub: http://github.com/still2009

import numpy as np
import keras.backend as K
from scipy.optimize import fsolve
from keras.metrics import mae as metric_mae
from keras.metrics import mse as metric_mse
from keras.regularizers import l1_l2, l1, l2

ENTROPY_MIN_CLIP = 1e-5


def random_entropy(loc_series, clip=True):
    Ni = np.unique(loc_series).shape[0]
    return np.clip(np.log2(Ni), ENTROPY_MIN_CLIP, None)


def shannon_entropy(loc_series, clip=True):
    u, counts = np.unique(loc_series, return_counts=True)
    Ni = u.shape[0]
    p = counts/loc_series.shape[0]
    return np.clip(-1 * np.sum(p * np.log2(p)), ENTROPY_MIN_CLIP, None)


def my_equal(prev, sub):
    n = prev.shape[0] + sub.shape[0]
    tmp = np.full((2, n), -1)
    tmp[0, :sub.shape[0]] = sub
    tmp[1, :prev.shape[0]] = prev
    x = max(prev.shape[0], sub.shape[0])
    return np.equal(tmp[0, :x], tmp[1, :x])


def real_entropy(loc_series, clip=True):
    """
    lempel ziv estimator
    :param loc_series: 
    :return: 
    """
    n = loc_series.shape[0]
    L = np.zeros(n)
    L[0]=1
    for i in range(1, n):
        prev, sub = loc_series[:i], loc_series[i]
        match = (prev == sub)
        if np.all(match==False):
            L[i]=1
        else:
            k=0
            while k<i:
                if i+k>n:
                    L[i]=0
                    break
                sub= loc_series[i:i + k + 1]
                for j in range(i):
                    match = my_equal(loc_series[j:j + len(sub)], sub)
                    if np.all(match == True):
                        break
                L[i]=len(sub)
                if not np.all(match==True):
                    k=i
                k += 1
    return np.clip(1/(1/n * np.sum(L))*np.log(n), ENTROPY_MIN_CLIP, None)


def max_pred(loc_series, entropy, init_value=0.9999):
    Ni = np.unique(loc_series).shape[0]
    if Ni == 1:
        return 1
    s = entropy(loc_series) if callable(entropy) else entropy
    # 求g=x+3, 再求x
    eq = lambda x: -1 * x * np.log2(x) - (1 - x) * np.log2(1 - x) \
                   + (1 - x) * np.log2(Ni - 1) - s
    return fsolve(eq, init_value)[0]


# np直接计算版RMSE
def np_rmse(y_true, y_pred):
    # flatted mean, the same as attconv paper
    return np.sqrt(np.mean(np.square(y_true-y_pred)))


def np_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# np直接计算版MAPE
def np_mape(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred) / np.clip(np.abs(y_true), 1, None)))


# np直接计算版修订mape
def np_loss(y_true, y_pred):
    """
    论文中的损失函数
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 修订版MAPE
    """
    diff = np.square(y_true-y_pred) / np.clip(np.square(y_true), 1, None)
    weight = 10
    return np.mean(np.square(y_true-y_pred) + weight * diff)


# 原版RMSE
def metric_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def metric_mape(y_true, y_pred):
    return K.mean(K.abs((y_true-y_pred) / K.clip(K.abs(y_true), 1, None)))


# 修订版MAPE，DMVST论文loss
def metric_loss(y_true, y_pred):
    """
    论文中的损失函数 
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 修订版MAPE
    """
    diff = K.square(y_true - y_pred) / K.clip(K.square(y_true), 1, None)
    weight = 10
    return K.mean(K.square(y_pred - y_true) + weight * diff)


LOSS_NAME_FUNC = {
        'mix': metric_loss,
        'rmse': metric_rmse,
        'mape': metric_mape,
        'mae': metric_mae,
        'mse': metric_mse
    }


def get_loss_func(name):
    assert name in LOSS_NAME_FUNC
    return LOSS_NAME_FUNC[name]

REGULAR_NAME_FUNC = {
    'l1': l1(),
    'l2': l2(),
    'l1_l2': l1_l2(),
    'none': None
}


def get_regularizer(name):
    assert name in REGULAR_NAME_FUNC
    return REGULAR_NAME_FUNC[name]
