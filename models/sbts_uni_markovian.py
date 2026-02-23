import numpy as np
import numba as nb
import datetime
import time


@nb.jit(nopython=True, cache=True)
def kernel(x, h):
    """Quartic compact-support kernel used for weighting neighbors.

    Parameters
    ----------
    x : np.ndarray
        Difference values between observations and the current state.
    h : float
        Bandwidth parameter.

    Returns
    -------
    np.ndarray
        Kernel weights with the same shape as ``x``.
    """
    return np.where(np.abs(x) < h, (h ** 2 - x ** 2) ** 2, 0)


@nb.jit(nopython=True, cache=True)
def simulate_kernel_mark(N, M, K, X, N_pi, h, deltati):
    """Simulate one univariate SBTS path with Markovian conditioning.

    Parameters
    ----------
    N : int
        Number of generated transitions; should satisfy ``N = X.shape[1] - 1``.
    M : int
        Number of reference trajectories in ``X``.
    K : int
        Markov order used in kernel reweighting.
    X : np.ndarray
        Reference trajectories with shape ``(M, N+1)``.
    N_pi : int
        Number of Euler substeps between two observations.
    h : float
        Kernel bandwidth.
    deltati : float
        Time interval between consecutive observations.

    Returns
    -------
    np.ndarray
        One generated path of shape ``(N+1,)`` including the initial point.
    """
    # Diffusion calendar
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    # Generate Brownian increments
    num_brownian = N * (len(v_time_step_Euler) - 1)
    Brownian = np.random.normal(0, 1, num_brownian)

    # Simulation initialization
    X_ = X[0, 0]
    timeSeries = np.zeros(N + 1)
    timeSeries[0] = X_

    weights = np.ones(M)
    weights_tilde = np.zeros(M)
    last_K = np.empty((K,), dtype=X.dtype)
    index_queue = 0
    index_ = 0

    # Simulation loop
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
                last_K[-1] = X_
            else:
                last_K[index_queue] = X_

            index_queue += 1
            weights[:] *= kernel(X[:, i] - X_, h)

        else:
            weights[:] = 1 / M

        weights_tilde[:] = weights[:] * np.exp(
            (X[:, i + 1] - X_) ** 2 / 2 * deltati
        )

        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]

            if k == 0:
                expec_den = np.sum(weights)
                expec_num = np.sum(weights * (X[:, i + 1] - X_))
            else:
                termtoadd = -(X[:, i + 1] - X_) ** 2 / (2 * (deltati - timeprev))
                termtoadd = weights_tilde * np.exp(termtoadd)
                expec_den = np.sum(termtoadd)
                termtoadd *= (X[:, i + 1] - X_)
                expec_num = np.sum(termtoadd)

            drift = (1 / (deltati - timeprev)) * (expec_num / expec_den) if expec_den > 0 else 0.0
            X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep)
            index_ += 1

        timeSeries[i + 1] = X_

    return timeSeries


def simusbts_mark(N, M, K, X, N_pi, h, deltati, M_simu):
    """Generate multiple univariate SBTS-Markovian trajectories.

    Parameters
    ----------
    N, M, K, X, N_pi, h, deltati :
        Same meaning as in :func:`simulate_kernel_mark`.
    M_simu : int
        Number of trajectories to generate.

    Returns
    -------
    np.ndarray
        Generated data of shape ``(M_simu, N)`` (initial point removed).
    """
    data_sb = np.zeros((M_simu, X.shape[-1]))
    st = datetime.datetime.now()
    print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
    time1 = time.perf_counter()

    for k in range(M_simu):
        data_sb[k] = simulate_kernel_mark(N, M, K, X, N_pi, h, deltati)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(
        f'Time with numba to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
        flush=True)
    return data_sb[:, 1:]
