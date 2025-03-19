import numpy as np


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min_, max_ = np.min(data, 0), np.max(data, 0)
    numerator = data - min_
    denominator = max_ - min_
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, max_, min_


def invert_back(norm_data, max_, min_):
    """Min Max inverse transform.

    Args:
      - data: normalized data

    Returns:
      - norm_data: original data
    """
    ori_data = norm_data * (max_ - min_)
    ori_data += min_
    return ori_data


def real_data_loading(ori_data, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
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
