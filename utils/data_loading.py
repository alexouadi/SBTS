import numpy as np


def MinMaxScaler(data):
    """Apply feature-wise min-max scaling.

    Parameters
    ----------
    data : np.ndarray
        Input array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Normalized data, per-feature maxima, and per-feature minima.
    """
    min_, max_ = np.min(data, 0), np.max(data, 0)
    numerator = data - min_
    denominator = max_ - min_
    norm_data = numerator / (denominator + 1e-7)
    return norm_data, max_, min_


def invert_back(norm_data, max_, min_):
    """Invert min-max normalization.

    Parameters
    ----------
    norm_data : np.ndarray
        Min-max normalized array.
    max_ : np.ndarray
        Per-feature maxima used during normalization.
    min_ : np.ndarray
        Per-feature minima used during normalization.

    Returns
    -------
    np.ndarray
        Data mapped back to the original scale.
    """
    ori_data = norm_data * (max_ - min_)
    ori_data += min_
    return ori_data


def real_data_loading(ori_data, seq_len):
    """Prepare chronological windowed sequences from raw tabular time series.

    Parameters
    ----------
    ori_data : np.ndarray
        Raw data with shape ``(time, features)``.
    seq_len : int
        Sliding-window sequence length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Shuffled windows of shape ``(n_windows, seq_len, features)``, maxima, and minima.
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
