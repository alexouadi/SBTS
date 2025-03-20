import numpy as np


def MinMaxScaler(data):
    """
    :params data: original data; [np.array]
    Returns: norm_data, normalized data; [np.array]
    """
    min_, max_ = np.min(data, 0), np.max(data, 0)
    numerator = data - min_
    denominator = max_ - min_
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, max_, min_


def invert_back(norm_data, max_, min_):
    """
    :params norm_data: normalized data; [np.array]
    :params max_: maximum value of the original data; [float]
    :params min_: minimum value of the original data; [float]
    Returns: original data; [np.array]
    """
    ori_data = norm_data * (max_ - min_)
    ori_data += min_
    return ori_data


def real_data_loading(ori_data, seq_len):
    """
    :params data: original data; [np.array]
    :params seq_len: sequence length; [int]
    Returns: preprocessed data; [np.array]
    """

    # Flip the data to make chronological data
    L, d = ori_data.shape
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data, max_, min_ = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = np.zeros((L - seq_len, seq_len, d))
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        temp_data[i] = ori_data[i:i + seq_len]

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    return temp_data[idx], max_, min_
