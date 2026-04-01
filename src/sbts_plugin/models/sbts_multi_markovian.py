import datetime
import time

import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True)
def kernel(x, h):
    """Quartic compact-support kernel used for vector-valued inputs.

    Parameters
    ----------
    x : np.ndarray
        Difference matrix with shape ``(n_samples, d)``.
    h : float
        Bandwidth parameter.

    Returns
    -------
    np.ndarray
        Kernel weights of shape ``(n_samples,)``.
    """
    x_norm = np.sqrt(np.sum(x**2, axis=1))
    return np.where(x_norm < h, (h**2 - x_norm**2) ** 2, 0)


@nb.jit(nopython=True, cache=True)
def simulate_kernel_vectorized_mark(N, M, d, K, X, N_pi, h, deltati):
    """Simulate one multivariate SBTS path with Markovian conditioning."""
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    num_brownian = (N * (len(v_time_step_Euler) - 1), d)
    Brownian = np.random.normal(0, 1, num_brownian)

    X_ = X[0, 0].copy()

    weights = np.ones(M)
    timeSeriesVector = np.zeros((N + 1, d))
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0
    index_ = 0

    for i in range(N):
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
        weights_tilde *= np.exp(np.sum(diff**2 / (2 * deltati), axis=1))

        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
            timestepsqrt = np.sqrt(timestep)

            if k == 0:
                expec_den = np.sum(weights)
                numerator = np.sum(weights[:, np.newaxis] * (X[:, i + 1] - X_), axis=0)
            else:
                diff = X[:, i + 1] - X_
                termtoadd = np.sum(diff**2, axis=1)
                termtoadd = weights_tilde * np.exp(-termtoadd / (2 * (deltati - timeprev)))
                expec_den = np.sum(termtoadd, axis=0)
                numerator = np.sum(termtoadd[:, np.newaxis] * (X[:, i + 1] - X_), axis=0)

            drift = (1 / (deltati - timeprev)) * (numerator / expec_den) if expec_den > 0 else np.zeros(d)
            X_ += drift * timestep + Brownian[index_] * timestepsqrt
            index_ += 1

        timeSeriesVector[i + 1, :] = X_

    return timeSeriesVector


@nb.jit(nopython=True, cache=True)
def sample_last_mark_multi(M, d, K, X, x_past, N_pi, h, deltati):
    """Sample one next point conditional on an observed multivariate context."""
    N = len(x_past)

    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    weights = np.ones(M)
    weights_tilde = np.ones(M)
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0

    for i in range(N):
        if i > 0:
            if index_queue >= K:
                X_oldest = last_K[0]
                kernel_oldest = kernel(X[:, i - K] - X_oldest, h)

                if np.any(kernel_oldest == 0):
                    weights = np.ones(M)
                    ind_ref = i - K
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
        weights_tilde *= np.exp(np.sum(diff**2 / (2 * deltati), axis=1))

    curr_time_step = N
    X_ = x_past[-1].copy()
    Brownian = np.random.normal(0, 1, (len(v_time_step_Euler) - 1, d))

    for k in range(len(v_time_step_Euler) - 1):
        timeprev = v_time_step_Euler[k]
        timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]

        if k == 0:
            expec_den = np.sum(weights)
            expec_num = np.sum(weights[:, np.newaxis] * (X[:, curr_time_step] - X_), axis=0)
        else:
            diff = X[:, curr_time_step] - X_
            termtoadd = np.sum(diff**2, axis=1)
            termtoadd = weights_tilde * np.exp(-termtoadd / (2 * (deltati - timeprev)))
            expec_den = np.sum(termtoadd, axis=0)
            expec_num = np.sum(termtoadd[:, np.newaxis] * diff, axis=0)

        drift = (1 / (deltati - timeprev)) * (expec_num / expec_den) if expec_den > 0 else np.zeros(d)
        X_ += drift * timestep + Brownian[k] * np.sqrt(timestep)

    return X_


def simusbts_multi_mark(N, M, d, K, X, N_pi, h, deltati, M_simu):
    """Generate multiple multivariate SBTS-Markovian trajectories."""
    data_sb = np.zeros((M_simu, X.shape[1], d))
    st = datetime.datetime.now()
    print(f"Start time: {st.strftime('%H:%M:%S')}", flush=True)
    time1 = time.perf_counter()

    for k in range(M_simu):
        data_sb[k, :] = simulate_kernel_vectorized_mark(N, M, d, K, X, N_pi, h, deltati)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f"Expected finish time: {st.strftime('%H:%M:%S')}", flush=True)

    print(f"Finish time: {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(
        f"Time with numba to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.",
        flush=True,
    )
    return data_sb[:, 1:]
