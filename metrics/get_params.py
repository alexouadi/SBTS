import numpy as np
import numba as nb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize


@nb.jit(nopython=True, cache=True)
def MLE_OU_robust(params, X, dt):
    """
    Compute the MLE on Ornstein-Uhlenbeck data.
    :params params: parameters to estimate; [list]
    :params X: time series data; [np.array]
    :params dt: time step; [float]
    return: negative log-likelihood over X; [float]
    """
    theta, mu, sigma = params
    N = len(X)
    logL = 0

    exp_neg_theta_dt = np.exp(-theta * dt)
    one_minus_exp_neg_theta_dt = 1 - exp_neg_theta_dt
    sigma_eta2 = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

    for t in range(N - 1):
        mu_t = X[t] * exp_neg_theta_dt + mu * one_minus_exp_neg_theta_dt
        residual = X[t + 1] - mu_t
        logL += -0.5 * np.log(2 * np.pi * sigma_eta2) - (residual ** 2) / (2 * sigma_eta2)

    return -logL


def plot_params_distrib_OU(X_data, X_sbts, dt, fix=False):
    """
    Plot the distribution of the estimated parameters on Ornstein-Uhlenbeck data.
    :params X_data: real data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params dt: time step; [float]
    :params fix: specified if the real parameters are fixed or not; [bool]
    """
    params_data = np.zeros((len(X_data), 3))
    for m in range(len(X_data)):
        params_init_data = [1, np.mean(X_data[m]), np.std(X_data[m])]
        result_data = minimize(MLE_OU_robust, np.array(params_init_data), args=(X_data[m], dt),
                               bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                               method='L-BFGS-B')
        params_data[m] = result_data.x

    params_sbts = np.zeros((len(X_sbts), 3))
    for m in range(len(X_sbts)):
        params_init_sbts = [1, np.mean(X_sbts[m]), np.std(X_sbts[m])]
        result_sbts = minimize(MLE_OU_robust, np.array(params_init_sbts), args=(X_sbts[m], dt),
                               bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                               method='L-BFGS-B')
        params_sbts[m] = result_sbts.x

    theta = np.random.uniform(.5, 2.5, 100000)
    mu = np.random.uniform(.5, 1.5, 100000)
    sigma = np.random.uniform(.1, .5, 100000)
    lines = [1.5, 1., 0.2]

    lower_bounds = np.percentile(params_data, 1, axis=0)
    upper_bounds = np.percentile(params_data, 99, axis=0)
    filtered_params_data = params_data[
        (params_data >= lower_bounds).all(axis=1) & (params_data <= upper_bounds).all(axis=1)]

    lower_bounds = np.percentile(params_sbts, 1, axis=0)
    upper_bounds = np.percentile(params_sbts, 99, axis=0)
    filtered_params_sbts = params_sbts[
        (params_sbts >= lower_bounds).all(axis=1) & (params_sbts <= upper_bounds).all(axis=1)]

    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    sns.kdeplot(ax=axs[0], data=params_data[:, 0], shade=True, label='Data')
    sns.kdeplot(ax=axs[0], data=params_sbts[:, 0], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[0], data=theta, shade=True, label='Real')
    else:
        line_obj = axs[0].axvline(x=lines[0], color='black', linestyle='--',
                                  label='Real')
        axs[0].legend(handles=[line_obj])
    axs[0].set_title(r'Distribution of params $\theta$')
    axs[0].legend()

    sns.kdeplot(ax=axs[1], data=filtered_params_data[:, 1], shade=True, label='Data')
    sns.kdeplot(ax=axs[1], data=filtered_params_sbts[:, 1], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[1], data=mu, shade=True, label='Real')
    else:
        line_obj = axs[1].axvline(x=lines[1], color='black', linestyle='--',
                                  label='Real')
        axs[1].legend(handles=[line_obj])
    axs[1].set_title(r'Distribution of params $\mu$')
    axs[1].legend()

    sns.kdeplot(ax=axs[2], data=params_data[:, 2], shade=True, label='Data')
    sns.kdeplot(ax=axs[2], data=params_sbts[:, 2], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[2], data=sigma, shade=True, label='Real')
    else:
        line_obj = axs[2].axvline(x=lines[2], color='black', linestyle='--',
                                  label='Real')
        axs[2].legend(handles=[line_obj])
    axs[2].set_title(r'Distribution of params $\sigma$')
    axs[2].legend()

    fig.tight_layout()
    plt.show()


@nb.jit(nopython=True, cache=True)
def MLE_BS_robust(params, log_returns, dt):
    """
    Compute the MLE on Black-Scholes data.
    :params params: parameters to estimate; [list]
    :params log_returns: time series data; [np.array]
    :params dt: time step; [float]
    return: negative log-likelihood over X; [float]
    """
    r, sigma = params

    logL = 0
    const = -0.5 * np.log(2 * np.pi * sigma ** 2 * dt)
    mu = (r - 0.5 * sigma ** 2) * dt
    den = 2 * sigma ** 2 * dt

    for t in range(len(log_returns)):
        logL += const - ((log_returns[t] - mu) ** 2) / den
    return -logL


def plot_params_distrib_BS(X_data, X_sbts, dt, fix=False):
    """
    Plot the distribution of the estimated parameters on Black-Scholes data.
    :params X_data: real data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params dt: time step; [float]
    :params fix: specified if the real parameters are fixed or not; [bool]
    """
    log_returns_data = X_data[:, 1:] - X_data[:, :-1]
    sigma_init = np.std(log_returns_data, axis=1)
    params_data = np.zeros((len(X_data), 2))

    for m in range(len(X_data)):
        result_data = minimize(
            MLE_BS_robust,
            x0=np.array([.01, sigma_init[m]]),
            args=(log_returns_data[m], dt),
            bounds=[(-np.inf, np.inf), (1e-6, np.inf)],
            method='L-BFGS-B'
        )
        params_data[m] = result_data.x

    log_returns_sbts = np.log(X_sbts[:, 1:] / X_sbts[:, :-1])
    sigma_init = np.std(log_returns_sbts, axis=1)
    params_sbts = np.zeros((len(X_sbts), 2))

    for m in range(len(X_sbts)):
        result_sbts = minimize(
            MLE_BS_robust,
            x0=np.array([.01, sigma_init[m]]),
            args=(log_returns_sbts[m], dt),
            bounds=[(-np.inf, np.inf), (1e-6, np.inf)],
            method='L-BFGS-B'
        )
        params_sbts[m] = result_sbts.x

    r = np.random.uniform(0.03, 0.3, 100000)
    sigma = np.random.uniform(0.1, 0.3, 100000)
    lines = [0.03, 0.1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot distribution of column 0
    sns.kdeplot(ax=axs[0], data=params_data[:, 0], shade=True, label='Data')
    sns.kdeplot(ax=axs[0], data=params_sbts[:, 0], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[0], data=r, shade=True, label='Real')
    else:
        line_obj = axs[0].axvline(x=lines[0], color='black', linestyle='--',
                                  label='Real')
        axs[0].legend(handles=[line_obj])
    axs[0].set_title(r'Distribution of params $r$')
    axs[0].legend()

    # Plot distribution of column 1
    sns.kdeplot(ax=axs[1], data=params_data[:, 1], shade=True, label='Data')
    sns.kdeplot(ax=axs[1], data=params_sbts[:, 1], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[1], data=sigma, shade=True, label='Real')
    else:
        line_obj = axs[1].axvline(x=lines[1], color='black', linestyle='--',
                                  label='Real')
        axs[1].legend(handles=[line_obj])

    axs[1].set_title(r'Distribution of params $\sigma$')
    axs[1].legend()

    fig.tight_layout()

    plt.show()


@nb.jit(nopython=True, cache=True)
def MLE_Heston_robust(params, X, dt):
    """
    Compute the MLE on Heston data.
    :params params: parameters to estimate; [list]
    :params X: time series data; [np.array]
    :params dt: time step; [float]
    return: negative log-likelihood over X; [float]
    """
    kappa, theta, xi, rho, r = params
    N = len(X)
    logL = 0.0

    S = X[:, 0]
    v = X[:, 1]
    for t in range(N - 1):
        S_t, S_t_next = S[t], S[t + 1]
        v_t = v[t]

        mu_S = np.log(S_t) + (r - 0.5 * v_t) * dt
        mu_v = v_t + kappa * (theta - v_t) * dt

        var_S = v_t * dt
        var_v = xi ** 2 * v_t * dt

        cov_Sv = rho * xi * v_t * dt
        cov_matrix = np.array([[var_S, cov_Sv], [cov_Sv, var_v]])

        if np.linalg.det(cov_matrix) <= 0:
            return 1e10
        inv_cov = np.linalg.inv(cov_matrix)
        det_cov = np.linalg.det(cov_matrix)

        joint_observation = np.array([
            np.log(S_t_next) - mu_S,
            v[t + 1] - mu_v
        ])
        joint_log_pdf = -0.5 * (
                2 * np.log(2 * np.pi) + np.log(det_cov) + joint_observation.T @ inv_cov @ joint_observation
        )
        logL -= joint_log_pdf

    return logL


def plot_params_distrib_Heston(X_data, X_sbts, dt, fix=False):
    """
    Plot the distribution of the estimated parameters on Black-Scholes data.
    :params X_data: real data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params dt: time step; [float]
    :params fix: specified if the real parameters are fixed or not; [bool]
    """
    params_data = np.zeros((len(X_data), 5))

    bounds = [
        (1e-6, None),  # kappa > 0
        (1e-6, None),  # theta > 0
        (1e-6, None),  # xi > 0
        (-1, 1),  # rho in [-1, 1]
        (-1, 1),  # r unrestricted
    ]

    for m in range(len(X_data)):
        result_data = minimize(
            MLE_Heston_robust,
            x0=np.array([2.0, 0.02, 0.1, -0.5, 0.03]),
            args=(X_data[m], dt),
            bounds=bounds,
            method='L-BFGS-B'
        )
        params_data[m] = result_data.x

    params_sbts = np.zeros((len(X_sbts), 5))
    for m in range(len(X_sbts)):
        result_sbts = minimize(
            MLE_Heston_robust,
            x0=np.array([2.0, 0.02, 0.1, -0.5, 0.03]),
            args=(X_sbts[m], dt),
            bounds=bounds,
            method='L-BFGS-B'
        )
        params_sbts[m] = result_sbts.x

    kappa = np.random.uniform(0.5, 1.5, 100000)
    theta = np.random.uniform(0.7, 1.3, 100000)
    xi = np.random.uniform(0.01, 0.9, 100000)
    rho = np.random.uniform(-0.9, 0.9, 100000)
    r = np.random.uniform(0.01, 0.3, 100000)

    params_real = [kappa, theta, xi, rho, r]
    labels = [r'$\kappa$', r'$\theta$', r'$\xi$', r'$\rho$', r'$r$']
    lines = [3., 1., .7, .7, .02]

    lower_bounds = np.percentile(params_data, 1, axis=0)
    upper_bounds = np.percentile(params_data, 99, axis=0)
    filtered_params_data = params_data[
        (params_data >= lower_bounds).all(axis=1) & (params_data <= upper_bounds).all(axis=1)]

    lower_bounds_sbts = np.percentile(params_sbts, 1, axis=0)
    upper_bounds_sbts = np.percentile(params_sbts, 99, axis=0)
    filtered_params_sbts = params_sbts[
        (params_sbts >= lower_bounds_sbts).all(axis=1) & (params_sbts <= upper_bounds_sbts).all(axis=1)]

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    for i, (param, label, line) in enumerate(zip(params_real, labels, lines)):
        sns.kdeplot(ax=axs[i // 3, i % 3], data=filtered_params_data[:, i], shade=True, label='Data')
        sns.kdeplot(ax=axs[i // 3, i % 3], data=filtered_params_sbts[:, i], shade=True, label='SBTS')
        if not fix:
            sns.kdeplot(ax=axs[i // 3, i % 3], data=param, shade=True, label='Real')
        else:
            line_obj = axs[i // 3, i % 3].axvline(x=line, color='black', linestyle='--',
                                                  label='Real')
            axs[i // 3, i % 3].legend(handles=[line_obj])
        axs[i // 3, i % 3].set_title(f'Distribution of param {label}')
        axs[i // 3, i % 3].legend()

    axs[1, 2].axis('off')
    fig.tight_layout()

    plt.show()
