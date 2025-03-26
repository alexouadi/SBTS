import numpy as np
import numba as nb
import datetime
import time


@nb.jit(nopython=True, cache=True)
def kernel(x, h):
    """
    Kernel function used for kernel regression.
    :params x: array of shape (len(x), d); [float]
    :params h: kernel bandwidth; [float]
    return: kernel function of shape (len(x),); [np.array]
    """
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    return np.where(x_norm < h, (h ** 2 - x_norm ** 2) ** 2, 0)


@nb.jit(nopython=True, cache=True)
def get_last_mark_multi(N, M, d, K, X, x_past, N_pi, h, deltati, itter):
    """
    Simulate the last point of a given time series via the Schrödinger Bridge kernel using markovian series.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples; [int]
    :params d: dimension of the time series; [int]
    :params K: markovian order of X; [int]
    :params X: samples of shape (M, N+1, d); [np.array]
    :params x_past: time series from which we simulate the last point, of shape(N, d); [np.array]
    :params N_pi: number of time steps in the Euler scheme; [int]
    :params h: kernel bandwidth; [float]
    :params deltati: time steps between two consecutive observations in the time series; [float]
    :params itter: number of path to generate; [int]
    return: mean of all last points generated; [np.array]
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
    """
    Simulate the last points of a given test via the Schrödinger Bridge kernel using markovian series.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples in the train set; [int]
    :params d: dimension of the time series; [int]
    :params K_markov: list of markovian order to test: [list]
    :params X: train samples set of shape (M, N+1, d); [np.array]
    :params x_past: test sample set from which we simulate the last points, of shape(len(x_past), N, d); [np.array]
    :params x_target: real last point of the test sample set series, of shape(len(x_past),); [np.array]
    :params N_pi: number of time steps in the Euler scheme; [int]
    :params h: list of kernel bandwidth to test; [list]
    :params deltati: time steps between two consecutive observations in the time series; [float]
    :params itter: number of path to generate; [int]
    return: the best windows found for all the h, along with their respective mse; [np.array, np.array]
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
