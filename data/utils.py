# _*_ coding : utf-8 _*_
# @Time : 2023/7/7 10:34
# @Author : Black
# @File : utils
# @Project : BabyBeatAnalyzer

import re
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence


def normalize(data, name='fhr'):
    """
    Normalize data
    :param name:
    :param data:
    :return:
    """
    # 用整个数据集的均值和标准差进行归一化，避免绝对大小的影响被消除

    # mean: 143.2439902761017
    # std: 13.578037051039761
    fhr_mean = 143.24
    fhr_std = 13.58

    # mean: 22.530914363844737
    # std: 17.208311089693304
    ucp_mean = 22.53
    ucp_std = 17.21

    if name == 'fhr':
        data = (data - fhr_mean) / fhr_std
    elif name == 'ucp':
        data = (data - ucp_mean) / ucp_std
    return data


def extract_integers(data):
    """
    Extract integers from a string
    :param data:
    :return:
    """
    integer_list = []
    for num in data:
        match = re.match(r'\d+', num)
        if match:
            integer_list.append(int(match.group()))

    return integer_list


def zero_detect(list):
    """
    Detect zero in a list
    :param list:
    :return:
    """
    # 检查是否存在0
    has_zero = 0 in list

    # 计算0的占比
    count_zero = list.count(0)
    zero_ratio = count_zero / len(list)

    print("Has zero:", has_zero)
    print("Zero count:", count_zero)
    print("Zero ratio:", zero_ratio)


def knn_fill(array, k):
    """
    KNN process for filling zero
    :param array:
    :param k:
    :return:
    """
    array = np.array(array)

    indices = np.where(array == 0)[0]  # 获取所有为0的索引
    array_copy = np.copy(array)  # 防止改变原数组

    for idx in indices:
        non_zero_indices = np.where(array_copy != 0)[0]  # 获取所有非0的索引
        # print('idx:', idx)
        # print('non_zero_indices:', non_zero_indices)
        distances = np.abs(non_zero_indices - idx)  # 计算与当前索引的距离
        # print(distances)
        nearest_indices = np.argsort(distances)[:k]  # 获取最近的k个非0索引
        # print(nearest_indices)
        mean_value = np.mean(array_copy[non_zero_indices[nearest_indices]])  # 计算平均值
        # print(mean_value)
        array_copy[idx] = int(mean_value)  # 取整并赋值给0位置

    return array_copy.tolist()


def preprocess(str, k=5, use_knn=True, use_zero_detect=False):
    """
    Preprocess data
    :param str:
    :param k:
    :param use_knn:
    :param use_zero_detect:
    :return:
    """
    data_str = str.split(',')
    data = extract_integers(data_str)
    if use_zero_detect:
        zero_detect(data)
    if use_knn:
        filled_data = knn_fill(data, k)
    return filled_data


def collate_fn(data):
    """
    Collate function
    :param data:
    :return:
    """
    X = []
    y = []
    # item: (data, label) tuple
    for item in data:
        X.append(item[0])
        y.append(item[1])
    X.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in X]
    X = pad_sequence(X, batch_first=True).float()
    X = X.unsqueeze(-1)
    X = pack_padded_sequence(X, seq_len, batch_first=True)
    return X, y
