import numpy as np


class GaussianProblem:
    def __init__(self, mu, sigma, mu_prior_mean, sigma_prior_mean, N, starting_mu, starting_sigma):
        self.starting_point = np.array([starting_mu, starting_sigma])
        self.d = 2
        self.dataset = np.random.normal(mu, sigma, N)
        self.mu_prior_mean = mu_prior_mean
        self.sigma_prior_mean = sigma_prior_mean
        self.N = N

    def pi_ratio(self, x, x_new):
        return np.exp(-0.5 * ((x_new[0] - self.mu_prior_mean) ** 2 - (x[0] - self.mu_prior_mean) ** 2) +
                      -0.5 * ((x_new[1] - self.sigma_prior_mean) ** 2 - (x[1] - self.sigma_prior_mean) ** 2))

    def gradient(self, x):
        m_1 = np.sum(self.dataset - x[0]) * 1.
        m_2 = np.sum(np.power((self.dataset - x[0]), 2)) * 1.
        mu = m_1 / x[1] ** 2
        sigma = -1. * self.N / x[1] \
                + m_2 / x[1] ** 3
        return np.array([mu, sigma])

    def Ginv(self, x):
        G = 1.0 * np.array([[1, 0], [0, 2]]) * self.N / x[1] ** 2
        Ginv = np.linalg.inv(G)
        return Ginv

    def grad_G(self, x):  # renvoie la liste des derivees de G en la ieme variable
        return [np.zeros((2, 2))] * 2
