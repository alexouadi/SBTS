from sklearn.preprocessing import StandardScaler
import numpy as np


def fix_dim(x):
    if x.ndim < 3:
        return x[:, :, np.newaxis]
    return x


def normalize_data(data):
    data = fix_dim(data)
    M, L, N = data.shape
    data_reshaped = data.reshape((M * L, N))  # Reshape to (ML, N)
    scaler = StandardScaler()
    normalized_data_reshaped = scaler.fit_transform(data_reshaped)
    normalized_data = normalized_data_reshaped.reshape((M, L, N))  # Reshape back to (M, L, N)
    return normalized_data, scaler


def invert_normalization(normalized_data, scaler):
    normalized_data = fix_dim(normalized_data)
    M, L, N = normalized_data.shape
    normalized_data_reshaped = normalized_data.reshape((M * L, N))
    inverted_data_reshaped = scaler.inverse_transform(normalized_data_reshaped)
    inverted_data = inverted_data_reshaped.reshape((M, L, N))
    return inverted_data


def min_max_data(data):
    m, M = data.min(axis=(0, 1)), data.max(axis=(0, 1))
    num = data - m
    den = M - m
    return num / den
