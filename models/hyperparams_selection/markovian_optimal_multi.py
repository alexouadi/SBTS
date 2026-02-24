import numpy as np
import numba as nb
import datetime
import time


@nb.jit(nopython=True, cache=True)
def kernel(x, h):
    """Quartic compact-support kernel used for vector-valued inputs.

    Parameters
    ----------
    x : np.ndarray
        Difference matrix with shape ``(n_samples, d)``.
    h : float
        Kernel bandwidth.

    Returns
    -------
    np.ndarray
        Kernel weights of shape ``(n_samples,)``.
    """
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    return np.where(x_norm < h, (h ** 2 - x_norm ** 2) ** 2, 0)


@nb.jit(nopython=True, cache=True)
def get_last_mark_multi(N, M, d, K, X, x_past, N_pi, h, deltati, itter):
    """Estimate the next point of one multivariate path with Markovian SBTS.

    Parameters
    ----------
    N, M, d, K, X, N_pi, h, deltati :
        SBTS settings, with ``X`` of shape ``(M, N+1, d)``.
    x_past : np.ndarray
        Observed past trajectory of shape ``(N, d)``.
    itter : int
        Number of Monte Carlo bridge paths used for averaging.

    Returns
    -------
    np.ndarray
        Mean predicted last value of shape ``(d,)``.
    """
    # Diffusion calendar
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    # Simulation initialization
    weights = np.ones(M)
    weights_tilde = np.ones(M)
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0

    for i in range(N):
        if i > 0:
            if index_queue >= K:
                X_oldest = last_K[0]
                kernel_oldest = kernel(X[:, i] - X_oldest, h)

                if np.any(kernel_oldest == 0):
                    weights = np.ones(M)
                    ind_ref = i - K + 1
                    for j in range(1, K):
                        weights *= kernel(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest
                
                last_K[:-1] = last_K[1:]
                last_K[-1] = x_past[i]

            else:
                last_K[index_queue] = x_past[i]

            index_queue += 1
            weights[:] *= kernel(X[:, i] - x_past[i], h)

        else:
            weights[:] = 1 / M

        weights_tilde = weights.copy()
        diff = X[:, i + 1] - x_past[i]
        weights_tilde *= np.exp(np.sum(diff ** 2 / (2 * deltati), axis=1))

    curr_time_step = i + 1
    X_last = np.zeros((itter, d))

    for itt in range(itter):
        X_ = x_past[-1]
        index_ = 0
        num_brownian = len(v_time_step_Euler) - 1
        Brownian = np.random.normal(0, 1, num_brownian)

        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]

            if k == 0:
                expec_den = np.sum(weights)
                expec_num = np.sum(weights[:, np.newaxis] * (X[:, curr_time_step] - X_), axis=0)
            else:
                diff = X[:, curr_time_step] - X_
                termtoadd = np.sum(diff ** 2, axis=1)
                termtoadd = weights_tilde * np.exp(-termtoadd / (2 * (deltati - timeprev)))
                expec_den = np.sum(termtoadd, axis=0)
                expec_num = np.sum(termtoadd[:, np.newaxis] * diff, axis=0)

            drift = (1 / (deltati - timeprev)) * (expec_num / expec_den) if expec_den > 0 else np.zeros(d)
            X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep)
            index_ += 1

        X_last[itt] = X_

    res = np.zeros(d)
    for i in range(d):
        res[i] = X_last[:, i].mean()

    return res


def get_optimal_order_multi(N, M, d, K_markov, X, x_past, x_target, N_pi, h, deltati, itter=50):
    """Search the best Markov order for multiple bandwidth values (multivariate case).

    Parameters
    ----------
    N, M, d, X, N_pi, deltati :
        SBTS settings, with ``X`` of shape ``(M, N+1, d)``.
    K_markov : list[int]
        Candidate Markov orders.
    x_past : np.ndarray
        Test past sequences with shape ``(n_test, N, d)``.
    x_target : np.ndarray
        True next values with shape ``(n_test, d)``.
    h : list[float]
        Candidate kernel bandwidths.
    itter : int, optional
        Number of bridge paths per prediction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ranked windows and associated MSE table (scaled by 100).
    """
    M_test = len(x_past)
    windows, mse_res = np.zeros((len(h), len(K_markov))), np.zeros((len(h), len(K_markov)))

    st = datetime.datetime.now()
    print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
    time1 = time.perf_counter()

    for i in range(len(h)):
        data_sb = np.zeros((len(K_markov), M_test, d))

        for k in range(len(K_markov)):
            for m in range(M_test):
                data_sb[k, m] = get_last_mark_multi(N, M, d, K_markov[k], X, x_past[m], N_pi, h[i], deltati, itter)

        mse = np.mean((data_sb - x_target) ** 2, axis=(1, 2))
        best_K = np.argsort(mse)
        windows[i], mse_res[i] = K_markov[best_K], mse * 100

        if i == 0:
            mm = (time.perf_counter() - time1) * (len(h) - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    ind = np.unravel_index(np.argmin(mse_res), mse_res.shape)
    print(f'Best sets of params: h={h[ind[0]]}, window={K_markov[ind[1]]}')
    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    return windows, np.round(mse_res, 4)
