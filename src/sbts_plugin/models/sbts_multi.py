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
def simulate_kernel_vectorized(N, M, d, X, N_pi, h, deltati):
    """Simulate one multivariate SBTS trajectory.

    Parameters
    ----------
    N : int
        Number of generated transitions; should satisfy ``N = X.shape[1] - 1``.
    M : int
        Number of reference trajectories in ``X``.
    d : int
        Feature dimension.
    X : np.ndarray
        Reference trajectories of shape ``(M, N+1, d)``.
    N_pi : int
        Number of Euler substeps per interval.
    h : float
        Kernel bandwidth.
    deltati : float
        Time interval between consecutive observations.

    Returns
    -------
    np.ndarray
        One generated path with shape ``(N+1, d)`` including the initial point.
    """
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    num_brownian = (N * (len(v_time_step_Euler) - 1), d)
    Brownian = np.random.normal(0, 1, num_brownian)

    X_ = X[0, 0].copy()

    weights = np.ones(M)
    timeSeriesVector = np.zeros((N + 1, d))
    index_ = 0

    for i in range(N):
        if i > 0:
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


def simulateSB_multi(N, M, d, X, N_pi, h, deltati, M_simu):
    """Generate multiple multivariate SBTS trajectories.

    Parameters
    ----------
    N, M, d, X, N_pi, h, deltati :
        Same meaning as in :func:`simulate_kernel_vectorized`.
    M_simu : int
        Number of trajectories to generate.

    Returns
    -------
    np.ndarray
        Generated data with shape ``(M_simu, N, d)`` (initial point removed).
    """
    data_sb = np.zeros((M_simu, X.shape[1], d))
    st = datetime.datetime.now()
    print(f"Start time: {st.strftime('%H:%M:%S')}", flush=True)
    time1 = time.perf_counter()

    for k in range(M_simu):
        data_sb[k, :] = simulate_kernel_vectorized(N, M, d, X, N_pi, h, deltati)
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
