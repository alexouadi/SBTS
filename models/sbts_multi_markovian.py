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
def simulate_kernel_vectorized_mark(N, M, d, K, X, N_pi, h, deltati):
    """
    Simulate 1 univariate time series via the Schrödinger Bridge kernel with markovian series.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples; [int]
    :params d: dimension of the time series; [int]
    :params K: markovian order; [int]
    :params X: samples of shape (M, N+1, d); [np.array]
    :params N_pi: number of time steps in the Euler scheme; [int]
    :params h: kernel bandwidth; [float]
    :params deltati: time steps between two consecutive observations in the time series; [float]
    return: 1 time series of shape (N+1,d); [np.array]
    """
    # Diffusion calendar
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    # Generate Brownian increments
    num_brownian = (N * (len(v_time_step_Euler) - 1), d)
    Brownian = np.random.normal(0, 1, num_brownian)

    # Simulation initialization
    X_ = X[0, 0].copy()

    weights = np.ones(M)
    timeSeriesVector = np.zeros((N + 1, d))
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0
    index_ = 0

    # Simulation loop
    for i in range(N):
        # Update weights and weights_tilde
        if i > 0:
            if index_queue >= K:
                X_oldest = last_K[0]
                kernel_oldest = kernel(X[:, i - K] - X_oldest, h)

                if np.any(kernel_oldest == 0):
                    weights = np.ones(M)
                    ind_ref = i - K
                    for j in range(1, K):
                        weights[:] *= kernel(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest

                last_K[:-1] = last_K[1:]
                last_K[-1] = X_

            else:
                last_K[index_queue] = X_
            index_queue += 1
            weights[:] *= kernel(X[:, i] - X_, h)

        else:
            weights[:] = 1 / M

        weights_tilde = weights.copy()
        diff = X[:, i + 1] - X_
        weights_tilde *= np.exp(np.sum(diff ** 2 / (2 * deltati), axis=1))

        # Perform steps within the interval
        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
            timestepsqrt = np.sqrt(timestep)

            if k == 0:
                expec_den = np.sum(weights)
                numerator = np.sum(weights[:, np.newaxis] * (X[:, i + 1] - X_), axis=0)
            else:
                diff = X[:, i + 1] - X_
                termtoadd = np.sum(diff ** 2, axis=1)
                termtoadd = weights_tilde * np.exp(-termtoadd / (2 * (deltati - timeprev)))
                expec_den = np.sum(termtoadd, axis=0)
                numerator = np.sum(termtoadd[:, np.newaxis] * (X[:, i + 1] - X_), axis=0)

            # Update X_
            drift = (1 / (deltati - timeprev)) * (numerator / expec_den) if expec_den > 0 else np.zeros(d)
            X_ += drift * timestep + Brownian[index_] * timestepsqrt
            index_ += 1

        # Store results
        timeSeriesVector[i + 1, :] = X_

    return timeSeriesVector


def simulateSB_multi_mark(N, M, d, K, X, N_pi, h, deltati, M_simu):
    """
    Simulate 1 univariate time series via the Schrödinger Bridge kernel with markovian series.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples; [int]
    :params d: dimension of the time series; [int]
    :params K: markovian order; [int]
    :params X: samples of shape (M, N+1, d); [np.array]
    :params N_pi: number of time steps in the Euler scheme; [int]
    :params h: kernel bandwidth; [float]
    :params deltati: time steps between two consecutive observations in the time series; [float]
    :params M_simu: number of time series to generate; [int]
    return: generated time series of shape (M_simu, N, d); [np.array]
    """
    data_sb = np.zeros((M_simu, X.shape[1], d))
    st = datetime.datetime.now()
    print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
    time1 = time.perf_counter()

    for k in range(M_simu):
        data_sb[k, :] = simulate_kernel_vectorized_mark(N, M, d, K, X, N_pi, h, deltati)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(
        f'Time with numba to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
        flush=True)
    return data_sb[:, 1:]
