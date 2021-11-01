# -*- coding: utf-8 -*-
# Author: Jack Lee
# GitHub: http://github.com/still2009

import datetime as dt
import functools
import hashlib
import json
import math
import os
import pickle
import re
import time
from collections import Iterable

import pandas as pd
import yaml
from gpuinfo import GPUInfo as gi
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler

from jack_utils.my_metrics import *


def avaiable_gpus(detect_time=3, cpu_ratio=0.5, mem_ratio=0.5):
    """
    get avaiable gpu device ids
    :param detect_time: seconds for detect
    :param use_ratio: ratio lower bound of cpu and mem usage
    :return: a list of avaiable gpu ids
    """
    assert type(detect_time) == int and cpu_ratio <= 1. and mem_ratio <= 1.
    # print('detecting valid gpus in %d seconds' % detect_time)

    # 1.使用正则表达式获取单GPU总内存(所有显卡相同的情况)
    total_mem = int(re.findall(r'([0-9]+)MiB \|', os.popen('nvidia-smi -i 0').readlines()[8])[0])

    # 2.检测符合资源要求的GPU
    pids, pcpu, mem, gpu_id = gi.get_info()
    for i in range(detect_time-1):
        time.sleep(1)
        pids, _pcpu, _mem, gpu_id = gi.get_info()
        _pcpu, _mem = np.asarray(_pcpu), np.asarray(_mem)
        pcpu = pcpu + _pcpu
        mem = mem + _mem
    pcpu, mem = np.asarray(pcpu) / detect_time, np.asarray(mem) / detect_time
    valid_gpus = np.argwhere((pcpu <= cpu_ratio * 100) & (mem <= mem_ratio * total_mem)).reshape(-1).tolist()
    valid_gpus = sorted(valid_gpus, key=lambda x: mem[x]*100 + pcpu[x])

    # 3. 打印信息
    # info = ['GPU%d: %d%%-%.1fG' % (x[0], x[1], x[2] / 1024) for x in zip(valid_gpus, pcpu[valid_gpus], mem[valid_gpus])]
    # print('valid gpus: | '.join(info))
    return valid_gpus


def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        delta = time.time() - start
        show = '{:.1f}min'.format(delta/60) if delta >= 60 else '{:.1f}s'.format(delta)
        print('func {} consumed {}'.format(func.__name__, show))
        return result

    return wrapper


def pk_save(obj, file_name):
    """
    保存对象，pickle.dump的快捷方式
    :param obj: 待保存的对象
    :param file_name: 保存的文件名
    :return: None
    """
    pickle.dump(obj, open(file_name, 'wb'))


def pk_load(file_name):
    """
    加载对象，pickle.load的快捷方式
    :param file_name: 源文件
    :return: obj
    """
    return pickle.load(open(file_name, 'rb'))


@time_it
def save_array(data, file_name):
    np.savez_compressed(file_name, data=data)


@time_it
def load_array(file_name):
    return np.load(file_name)['data']


# 进行MinMaxScale
def scale(scaler, data):
    if data.ndim != 2:
        return scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
    else:
        return scaler.transform(data)


def unscale(scaler, data):
    if data.ndim != 2:
        return scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)
    else:
        return scaler.inverse_transform(data)


def data_split(data, condition):
    """
        train/val/test数据集切分
    :param data: ndarray like obj, 按照第一维切分 
    :param condition: 切分的条件，可以为切分比例或者三元列表
    :return: NamedDict
    """
    if isinstance(condition, float):
        ratio = condition
        assert 0. < ratio < 1
        n_samples = data.shape[0]
        n_test = int(n_samples * (1 - ratio))
        n_train = int((n_samples - n_test) * ratio)
        n_val = n_samples - n_test - n_train
    else:
        n_train, n_val, n_test = condition
        assert sum(condition) == data.shape[0]
    return NamedDict({
        'train': data[:n_train],
        'val': data[n_train: n_train + n_val],
        'test': data[-1 * n_test:]
    })


def get_scaler(data, scale_range):
    scaler = MinMaxScaler(feature_range=scale_range)
    if np.ndim(data) > 2:
        scaler.fit(data.reshape(-1, 1))
    else:
        scaler.fit(data)
    return scaler


def generate_data_slide(
        time_series,
        x_window_size,  # x_seq_len
        x_stride=1,  # x滑动的步长
        y_window_size=1,  # y_seq_len
        xy_distance=None,  # y与x的距离
        gen_time=False,  # 是否以时间作为特征
        time_range=(0, 1)  # 时间的归一化范围
):
    """
    输入n维时间序列，输出n+1维滑动窗口切片样本集合
    shapes:
        in(steps, dim2, dim3) -> out(steps-window_size-1, window_size, dim2, dim3)
    """
    if xy_distance is None:
        xy_distance = x_window_size

    # 1.滑窗生成数据
    data_x, data_y = [], []
    N = time_series.shape[0]
    t = np.linspace(time_range[0], time_range[1], N)
    assert xy_distance >= 1
    for i in range(0, N-max(xy_distance + y_window_size, x_window_size), x_stride):
        if not gen_time:
            data_x.append(time_series[i: i + x_window_size])
            data_y.append(time_series[i + xy_distance: i + xy_distance + y_window_size])
        else:
            data_x.append(t[i: i + x_window_size])
            data_y.append(time_series[i + xy_distance: i + xy_distance + y_window_size])
    data_x, data_y = np.stack(data_x, axis=0), np.stack(data_y, axis=0)
    return data_x, data_y


def now_str():
    now = dt.datetime.now()
    return now.strftime('%Y%m%d_%H%M%S')


def param_tuple(func):
    @functools.wraps(func)
    def wrapper(args):
        return func(*args)

    return wrapper


def lnglat2mile(lat):
    """
    该经纬度上1米所代表的实际米数
    :param lat: 纬度值
    :return: 1米所占经度的度数，1米所占纬度的度数
    """
    earth_perimeters = 2 * math.pi * 6371000
    lng_perimeters = 40030173
    lat_perimeters = earth_perimeters * math.cos(lat)
    return 360 / lat_perimeters, 360 / lng_perimeters


path_join = os.path.join
hasnan = lambda x: np.isnan(x).any()


def cartesian_product(a, b):
    # a,b must be 1D array
    return np.stack([np.repeat(a, b.shape[0]).T, np.tile(b, a.shape[0]).T], axis=1)


def count_params(keras_model):
    return np.sum([K.count_params(p) for p in keras_model.trainable_weights])


class NamedDict:
    """
    字典dict的包装类, 提供dict.key的属性访问方式
    可以嵌套NamedDict
    """

    def __init__(self, data=None, **kwargs):
        if data is not None:
            assert isinstance(data, dict)
            self.__data__ = data
        else:
            self.__data__ = kwargs

    def get_raw(self):
        return self.__data__

    @staticmethod
    def from_yaml(fname):
        data = None
        with open(fname, 'r') as f:
            data = yaml.load(f)
        if data is None:
            raise ValueError('Cannot be None')
        return NamedDict(data=data)

    def to_yaml(self, fname):
        with open(fname, 'w') as f:
            yaml.dump(self.__data__, f)
            print('yaml file dumped')

    @staticmethod
    def from_json(fname):
        data = None
        with open(fname, 'r') as f:
            data = json.load(f)
        if data is None:
            raise ValueError('Cannot be None')
        return NamedDict(data=data)

    def to_json(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.__data__, f)
            print('json file dumped')

    def __getattr__(self, item):
        if item is '__data__':
            super(NamedDict, self).__getattribute__(item)
        elif item in self.__data__:
            result = self.__data__.get(item)
            if isinstance(result, dict) and not isinstance(result, NamedDict):
                return NamedDict(data=result)
            return result
        else:
            try:
                x = self.__data__.__getattribute__(item)
                return x
            except AttributeError as e:
                # print(e)
                raise AttributeError('NamedDict has no attribute %s' % item)

    def __setattr__(self, key, value):
        if key is '__data__':
            super(NamedDict, self).__setattr__(key, value)
        elif key in self.__data__:
            self.__data__[key] = value
        else:
            try:
                self.__data__.__setattr__(key, value)
            except AttributeError as e:
                # print(e)
                raise AttributeError('NamedDict has no attribute %s' % key)

    __getitem__ = __getattr__

    def __repr__(self):
        return str(self.__data__)

    __str__ = __repr__

    def __iter__(self):
        return self.__data__.__iter__()

    @staticmethod
    def sorted_flatten(value, key=None, it2str=False):
        result = {}
        if (not hasattr(value, 'keys')) and (not hasattr(value, 'values')):
            if isinstance(value, Iterable) and it2str and (not isinstance(value, str)):
                result[key] = '-'.join(str(x) for x in value)
            else:
                result[key] = value
            return result
        else:
            flatten_values = {}
            for k in sorted(value.keys()):
                key_prefix = '{}_{}'.format(key, k) if key is not None else k
                flatten_values.update(NamedDict.sorted_flatten(value[k], key_prefix, it2str))
            return flatten_values

    def sorted_str(self):
        flated = self.sorted_flatten(self.get_raw())
        return '\n'.join(['{}: {}'.format(k, flated[k]) for k in sorted(flated.keys())])

    def sha1(self):
        return hashlib.sha1(self.sorted_str().encode('utf8')).digest()

    def equal(self, another):
        assert isinstance(another, NamedDict)
        return self.sha1() == another.sha1()

    def in_set(self, target_set):
        for obj in target_set:
            if self.equal(obj):
                return True
        return False

    @staticmethod
    def diff(one, other):
        a, b = NamedDict.sorted_flatten(one), NamedDict.sorted_flatten(other)
        info_a, info_b = {}, {}
        inter_keys = a.keys() & b.keys()
        for k in inter_keys:
            if a[k] != b[k]:
                info_a[k], info_b[k] = a[k], b[k]
        res_a = {k: a[k] for k in (a.keys() - inter_keys)}
        info_a.update(res_a)

        res_b = {k: b[k] for k in (b.keys() - inter_keys)}
        info_b.update(res_b)
        return info_a, info_b

    @staticmethod
    def inter(one, other):
        a, b = NamedDict.sorted_flatten(one), NamedDict.sorted_flatten(other)
        inter_keys = a.keys() & b.keys()
        return {k: a[k] for k in inter_keys}

    @staticmethod
    def set_inter(a, b):
        # 求两个NamedDict collection的交集，并按hash顺序返回list
        dict_a = {o.sha1(): o for o in a}
        dict_b = {o.sha1(): o for o in b}
        inter = dict_a.keys() & dict_b.keys()
        return [dict_a[k] for k in sorted(inter)]

    @staticmethod
    def set_sub(a, b):
        # 求两个NamedDict collection的差集，并按hash顺序返回list
        dict_a = {o.sha1(): o for o in a}
        dict_b = {o.sha1(): o for o in b}
        subtract = dict_a.keys() - dict_b.keys()
        return [dict_a[k] for k in sorted(subtract)]

    @staticmethod
    def set_unique(target):
        # 对NamedDict collection去重，并按hash顺序返回list
        dict_data = {o.sha1(): o for o in target}
        return [dict_data[k] for k in sorted(dict_data.keys())]


class UnscaledMetrics(Callback):  # todo 性能提升，使用tensor计算metric而不是numpy
    def __init__(self, scaler, n_inputs, node_index=None, silence=True):
        """
        :param n_inputs: 训练时输入张量的个数(考虑具有多个输入的网络)
        :param scaler: 数据缩放对象
        :param node_index: 需要衡量的指定位置的网格索引ndarray, 可选
        """
        assert hasattr(scaler, 'inverse_transform')
        assert hasattr(scaler, 'transform')
        super(UnscaledMetrics, self).__init__()
        self.scaler = scaler
        self.node_index = node_index
        self.silence = silence
        self.n_inputs = n_inputs
        if node_index is not None:
            print('there are %d sp_nodes' % node_index.shape[0])

        self.metrics = NamedDict({
            'loss': [],
            'loss_sn': [],
            'rmse': [],
            'rmse_sn': [],
            'mape': [],
            'mape_sn': [],
            'mae': [],
            'mae_sn': [],
        })
        self.best_epoch_all = 1
        self.best_epoch_sp = 1

    def on_epoch_end(self, epoch, logs=None):
        """
        compute metrics for all nodes and specific nodes
        :param epoch: current epoch 
        """
        x, y = self.validation_data[:self.n_inputs], self.validation_data[self.n_inputs]
        y = unscale(self.scaler, y)

        # 1.当使用grid repr/ graph repr时，评估所有节点
        _start = time.time()
        y_pred = unscale(self.scaler, np.nan_to_num(self.model.predict(x), False))
        loss, rmse, mape, mae = np_loss(y, y_pred), np_rmse(y, y_pred), np_mape(y, y_pred), np_mae(y, y_pred)
        if not self.silence:
            now = time.time()
            print(' - %ds - val_loss: %.4f, val_rmse: %.4f, val_mape:%.4f, val_mae:%.4f - all_nodes' %
                  (now-_start, loss, rmse, mape, mae))
        self.metrics.loss.append(loss)
        self.metrics.rmse.append(rmse)
        self.metrics.mape.append(mape)
        self.metrics.mae.append(mae)
        if loss <= self.metrics.loss[self.best_epoch_all - 1]:
            self.best_epoch_all = epoch

        # 2.当使用grid repr时，评估选定节点
        if self.node_index is not None:
            _start = time.time()
            if np.ndim(y) == 4:
                y_sn = y[:, self.node_index[:, 0], self.node_index[:, 1]]
                y_pred_sn = y_pred[:, self.node_index[:, 0], self.node_index[:, 1]]
            if np.ndim(y) == 5:
                y_sn = y[:, :, self.node_index[:, 0], self.node_index[:, 1]]
                y_pred_sn = y_pred[:, :, self.node_index[:, 0], self.node_index[:, 1]]
            loss_sn = np_loss(y_sn, y_pred_sn)
            rmse_sn, mape_sn, mae_sn = np_rmse(y_sn, y_pred_sn), np_mape(y_sn, y_pred_sn), np_mae(y_sn, y_pred_sn)
            if not self.silence:
                now = time.time()
                print(' - %ds - val_loss: %.4f, val_rmse: %.4f, val_mape:%.4f, val_mae:%.4f - sp_nodes' %
                      (now-_start, loss_sn, rmse_sn, mape_sn, mae_sn))
            self.metrics.loss_sn.append(loss_sn)
            self.metrics.rmse_sn.append(rmse_sn)
            self.metrics.mape_sn.append(mape_sn)
            self.metrics.mae_sn.append(mae_sn)
            if loss_sn <= self.metrics.loss_sn[self.best_epoch_sp - 1]:
                self.best_epoch_sp = epoch

    def on_train_end(self, logs=None):
        # 打印分数误差最低的epoch的参数
        ep = self.best_epoch_all - 1
        if ep > 0:
            print('best epoch_all %d:  val_loss: %.4f, val_rmse: %.4f, val_mape:%.4f, val_mae:%.4f' %
                  (ep, self.metrics.loss[ep], self.metrics.rmse[ep], self.metrics.mape[ep], self.metrics.mae[ep])
                  )
        ep = self.best_epoch_sp
        if self.node_index is not None:
            print('best epoch_sp %d:  val_loss: %.4f, val_rmse: %.4f, val_mape:%.4f, val_mae:%.4f' %
                  (ep, self.metrics.loss_sn[ep], self.metrics.rmse_sn[ep], self.metrics.mape_sn[ep], self.metrics.mae_sn[ep])
                  )

    def summary(self):
        info = {k: self.metrics[k] for k in self.metrics if len(self.metrics[k]) > 0}
        return pd.DataFrame(info)


if __name__ == '__main__':
    x = NamedDict.from_yaml('conf.v3.yaml')
    x.to_yaml('test.yaml')
    y = NamedDict.from_yaml('test.yaml')
    print(x == y, x.equal(y))