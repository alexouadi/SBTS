import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd

from discriminative_score import *
from predictive_score import *


def min_max_data(data):
    """Apply feature-wise min-max scaling across batch and time axes.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape ``(batch, time, features)``.

    Returns
    -------
    np.ndarray
        Scaled array in ``[0, 1]`` per feature.
    """
    m, M = data.min(axis=(0, 1)), data.max(axis=(0, 1))
    num = data - m
    den = M - m
    return num / den


def fix_dim(x):
    """Ensure input data has three dimensions: ``(batch, time, features)``."""
    if x.ndim < 3:
        return x[:, :, np.newaxis]
    return x


def plot_sample(X_data, X_sbts, x0=0):
    """Plot random univariate trajectories from real and generated datasets.

    Parameters
    ----------
    X_data : np.ndarray
        Original data with shape ``(M, N)``.
    X_sbts : np.ndarray
        Generated data with shape ``(M_simu, N)``.
    x0 : float, optional
        Initial value prepended for display.
    """
    N = X_data.shape[-1]
    x_d, x_s = np.zeros((X_data.shape[0], N + 1)), np.zeros((X_sbts.shape[0], N + 1))
    x_d[:, 0], x_s[:, 0] = x0, x0
    x_d[:, 1:], x_s[:, 1:] = X_data, X_sbts
    X_data, X_sbts = x_d, x_s

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(4):
        j1 = np.random.randint(len(X_data))
        j2 = np.random.randint(len(X_sbts))
        ax[0].plot(X_data[j1], linewidth=1.5)
        ax[1].plot(X_sbts[j2], linewidth=1.5)

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Data')
    ax[0].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('time')
    ax[1].set_ylabel('SBTS')
    ax[1].tick_params(axis='both', which='major', labelsize=13)
    plt.show()


def plot_sample_multi(X_data, X_sbts, col=None, x0=1):
    """Plot one random multivariate trajectory from each dataset.

    Parameters
    ----------
    X_data : np.ndarray
        Original data with shape ``(M, N, d)``.
    X_sbts : np.ndarray
        Generated data with shape ``(M_simu, N, d)``.
    col : list[str] | None, optional
        Feature names used in the legend.
    x0 : float, optional
        Initial value prepended for display.
    """
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    B, N, d = X_sbts.shape
    x_d, x_s = np.zeros((X_data.shape[0], N + 1, d)), np.zeros((X_sbts.shape[0], N + 1, d))
    x_d[:, 0], x_s[:, 0] = x0, x0
    x_d[:, 1:], x_s[:, 1:] = X_data, X_sbts
    X_data, X_sbts = x_d, x_s

    ind_data = np.random.randint(len(X_data))
    ind_sbts = np.random.randint(len(X_sbts))

    for i in range(X_data.shape[-1]):
        if col is not None:
            ax[0].plot(X_data[:, :, i][ind_data], linewidth=1.5, label=col[i])
            ax[1].plot(X_sbts[:, :, i][ind_sbts], linewidth=1.5, label=col[i])
        else:
            ax[0].plot(X_data[:, :, i][ind_data], linewidth=1.5)
            ax[1].plot(X_sbts[:, :, i][ind_sbts], linewidth=1.5)

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Data')
    ax[0].legend()
    ax[0].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('time')
    ax[1].set_ylabel('SBTS')
    ax[1].legend()
    ax[1].tick_params(axis='both', which='major', labelsize=13)


def get_scores(X_data, X_sbts, col_pred=None, itt=2000, n_temp=10, min_max=False, device=torch.device('cpu'), device_ids=[2]):
    """Compute discriminative and predictive metrics used in the notebook demo.

    Parameters
    ----------
    X_data : np.ndarray
        Original dataset.
    X_sbts : np.ndarray
        Generated dataset.
    col_pred : int | None, optional
        Feature index used as the prediction target. Defaults to the last feature.
    itt : int, optional
        Training iterations per metric run.
    n_temp : int, optional
        Number of repeated evaluations.
    min_max : bool, optional
        If ``True``, apply min-max scaling before predictive scoring.
    device : torch.device, optional
        Device used during model training.
    device_ids : list[int], optional
        GPU IDs for discriminative score when using multi-GPU setups.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Discriminative scores and predictive scores, each of length ``n_temp``.
    """
    X_data, X_sbts = fix_dim(X_data), fix_dim(X_sbts)
    data = torch.tensor(X_data).to(torch.float32).to(device)
    data_g = torch.tensor(X_sbts).to(torch.float32).to(device)

    if min_max:
        X_data_scaled, X_sbts_scaled = min_max_data(X_data), min_max_data(X_sbts)
        data_scaled = torch.tensor(X_data_scaled).to(torch.float32).to(device)
        data_g_scaled = torch.tensor(X_sbts_scaled).to(torch.float32).to(device)
    
    if col_pred is None:
        col_pred = data.shape[-1] - 1  # by default

    disc_score, pred_score = np.zeros(n_temp), np.zeros(n_temp)
    st = datetime.datetime.now()
    print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
    time1 = time.perf_counter()

    for i in range(n_temp):

        if len(data) >= len(data_g):
            idx = np.random.permutation(len(data))
            disc_score[i] = discriminative_score_metrics(data[idx[:len(data_g)]], data_g, itt, device, device_ids)
            if min_max:
                pred_score[i] = predictive_score_metrics(data_scaled[idx[:len(data_g_scaled)]], data_g_scaled, col_pred, itt, device=device)
            else:
                pred_score[i] = predictive_score_metrics(data[idx[:len(data_g)]], data_g, col_pred, itt, device=device)

        else:
            idx = np.random.permutation(len(data_g))
            disc_score[i] = discriminative_score_metrics(data, data_g[idx[:len(data)]], itt, device, device_ids)
            if min_max:
                pred_score[i] = predictive_score_metrics(data_scaled, data_g_scaled, col_pred, itt, device=device)
            else:
                pred_score[i] = predictive_score_metrics(data, data_g, col_pred, itt, device=device)

        if i == 0:
            mm = (time.perf_counter() - time1) * (n_temp - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(
        f'Discriminative score (lower the better): {np.round(disc_score.mean(), 3)} +- {np.round(disc_score.std(), 3)}')
    print(f'Predictive score (lower the better): {np.round(pred_score.mean(), 3)} +- {np.round(pred_score.std(), 3)}')

    return disc_score, pred_score


def get_stats(X_data, X_sbts, col=None):
    """Summarize distributional statistics for real vs generated data.

    Parameters
    ----------
    X_data : np.ndarray
        Original dataset.
    X_sbts : np.ndarray
        Generated dataset.
    col : list[str] | None, optional
        Feature names used as index labels.

    Returns
    -------
    pd.DataFrame
        Table containing percentiles, mean, standard deviation, min and max.
    """

    X_data, X_sbts = fix_dim(X_data), fix_dim(X_sbts)

    # calculate 1% and 99% percentiles for both arrays
    percentiles1 = np.percentile(X_data, [1, 99], axis=(0, 1))  # shape (2, D)
    lower_percentile1 = percentiles1[0, :]  # shape (D,)
    upper_percentile1 = percentiles1[1, :]  # shape (D,)

    percentiles2 = np.percentile(X_sbts, [1, 99], axis=(0, 1))  # shape (2, D)
    lower_percentile2 = percentiles2[0, :]  # shape (D,)
    upper_percentile2 = percentiles2[1, :]  # shape (D,)

    # calculate mean and standard deviation for both arrays
    mean1 = np.mean(X_data, axis=(0, 1))  # shape (D,)
    std1 = np.std(X_data, axis=(0, 1))  # shape (D,)

    mean2 = np.mean(X_sbts, axis=(0, 1))  # shape (D,)
    std2 = np.std(X_sbts, axis=(0, 1))  # shape (D,)

    min_data = X_data.min(axis=(0, 1))
    min_sbts = X_sbts.min(axis=(0, 1))

    max_data = X_data.max(axis=(0, 1))
    max_sbts = X_sbts.max(axis=(0, 1))

    if col is None:
        col = range(len(lower_percentile1))

    df = pd.DataFrame({
        'Feature': col,
        '1% Data': lower_percentile1,
        '1% SBTS': lower_percentile2,
        '99% Data': upper_percentile1,
        '99% SBTS': upper_percentile2,
        'Mean Data': mean1,
        'Mean SBTS': mean2,
        'Std Data': std1,
        'Std SBTS': std2,
        'Min Data': min_data,
        'Min SBTS': min_sbts,
        'Max Data': max_data,
        'Max SBTS': max_sbts
    })

    df.set_index('Feature', inplace=True)
    return df.round(3)
