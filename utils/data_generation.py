import numpy as np


class Generate_Data:
    def __init__(self, M):
        """
        M: number of time series to generate -> [int]
        """

        self.M = M

    def generate_GARCH(self, N, alpha_0=5, alpha_1=0.4, alpha_2=0.1, s=0.1, x0=0):
        """
        N: length of time series to generate -> [int] : default = 60
        s: variance of the noise -> [float] : default = 0.1
        alpha0, alpha1, alpha2: model parameters : -> [float]
        """

        def simulate():
            time_series = list()
            x_next = 0.0
            x_prev = 0.0

            for t in range(N + 50):
                if t >= 50:
                    time_series.append(x_next)

                sigma = np.sqrt(alpha_0 + alpha_1 * x_next ** 2 + alpha_2 * x_prev ** 2)
                x_prev = x_next
                x_next = sigma * np.random.normal(scale=s)

            return time_series

        X = np.array([simulate() for i in range(self.M)])
        X_garch = np.zeros((self.M, N + 1))
        X_garch[:, 0], X_garch[:, 1:] = x0, X
        return X_garch

    def generate_OU(self, theta_range, mu_range, sigma_range, N, dt=1 / 252, x0=1):
        """
        theta_range: range of theta values -> [list of two ints]
        sigma_range: range of sigma values -> [list of two ints]
        mu_range: range of mu values -> [list of two ints]
        N: time series length -> [int]
        dt: time step -> [float] : default = 0.01
        X0: initial value -> [float] : default = 1
        """

        def simulate(theta, mu, sigma):
            X = np.zeros(N + 1)
            X[0] = x0
            for t in range(1, N + 1):
                mu_t = X[t - 1] * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt))
                sigma_t = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
                X[t] = mu_t + np.sqrt(sigma_t) * np.random.normal(0, 1)
            return X

        thetas = np.random.uniform(theta_range[0], theta_range[1], self.M)
        mus = np.random.uniform(mu_range[0], mu_range[1], self.M)
        sigmas = np.random.uniform(sigma_range[0], sigma_range[1], self.M)
        return np.array([simulate(thetas[i], mus[i], sigmas[i]) for i in range(self.M)])

    def generate_Heston(self, r_range, kappa_range, theta_range, rho_range, xi_range, N, dt=1 / 252, S0=1, v0=1):
        """
        r/k/theta/pho/xi_range: range of params values -> [list of two ints]
        N: time series length -> [int]
        T: terminal time -> [int] : default = 1
        S0: price initial value -> [float] : default = 1
        v0: price initial value -> [float] : default = 0
        """

        heston = np.zeros((self.M, N + 1, 2))

        def simulate(r, kappa, theta, rho, xi, dt=dt):
            prices, vol = np.zeros(N + 1), np.zeros(N + 1)
            S_t, v_t = S0, v0
            prices[0], vol[0] = S0, v0
            for t in range(1, N + 1):
                WT = np.random.multivariate_normal(np.array([0, 0]),
                                                   cov=np.array([[1, rho],
                                                                 [rho, 1]])) * np.sqrt(dt)

                S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[0]))
                v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[1])
                prices[t] = S_t
                vol[t] = v_t

            return prices[:, np.newaxis], vol[:, np.newaxis]

        r = np.random.uniform(r_range[0], r_range[1], self.M)
        kappa = np.random.uniform(kappa_range[0], kappa_range[1], self.M)
        theta = np.random.uniform(theta_range[0], theta_range[1], self.M)
        rho = np.random.uniform(rho_range[0], rho_range[1], self.M)
        xi = np.random.uniform(xi_range[0], xi_range[1], self.M)

        for i in range(self.M):
            price, vol = simulate(r[i], kappa[i], theta[i], rho[i], xi[i])
            serie = np.concatenate([price, vol], axis=1)
            heston[i] = serie

        return heston

    def generate_sine(self, N, d, x0=0):

        data = np.zeros((self.M, N + 1, d))
        data[:, 0] = x0

        for i in range(self.M):
            for k in range(d):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                temp_data = [np.sin(freq * j + phase) for j in range(N)]
                data[i, 1:, k] = temp_data
            data[i] = (data[i] + 1) * 0.5

        return data

    def generate_AR_multi(self, N, d, phi, sigma, x0=0):
        """
        Generate sequences from an autoregressive multivariate Gaussian model.
        Parameters:
            phi (float): Autoregressive coefficient in [0, 1].
            sigma (float): Controls the correlation across features, in [-1, 1].
            d (int): Number of features.
            N (int): Number of time steps to generate.
            x0: initial value
        """

        data = np.zeros((self.M, N + 1, d))
        data[:, 0] = x0

        sigma_matrix = sigma * np.ones((d, d)) + (1 - sigma) * np.eye(d)
        noise = np.random.multivariate_normal(mean=np.zeros(d), cov=sigma_matrix, size=(self.M, N))
        for t in range(1, N + 1):
            data[:, t] = phi * data[:, t - 1] + noise[:, t - 1]

        return data
