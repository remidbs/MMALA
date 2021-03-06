import numpy as np


class MALAGaussianOtherParametrization:
    def __init__(self, mu, sigma, N, eps, prior_mu, prior_sigma):
        self.eps = eps
        self.dataset = np.random.normal(loc=mu, scale=sigma, size=N)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def log_q(self, mu1, sigma1, mu2, sigma2):
        N = len(self.dataset)
        m_1 = np.sum(self.dataset - mu2)
        m_2 = np.sum(np.power((self.dataset - mu2), 2))
        eps = self.eps

        theta = np.array([mu1, sigma1])

        mu = np.array(
            [mu2 + eps ** 2 * m_1 * 0.5 / sigma2 ** 4,
             sigma2 + eps ** 2 * 2 * m_2 / sigma2 ** 5 - 2 * eps ** 2 * N / sigma2])
        return -0.5 / eps ** 2 * (theta - mu).T.dot(theta - mu)

    def acceptation_rate(self, mu, sigma, new_mu, new_sigma):
        prior_ratio = np.exp(-0.5 * ((new_mu - self.prior_mu) ** 2 - (mu - self.prior_mu) ** 2) +
                             -0.5 * ((new_sigma - self.prior_sigma) ** 2 - (sigma - self.prior_sigma) ** 2))
        jump_ratio = np.exp(self.log_q(mu, sigma, new_mu, new_sigma) - self.log_q(new_mu, new_sigma, mu, sigma))
        return min(1, prior_ratio * jump_ratio)

    def iteration(self, mu, sigma):
        N = len(self.dataset)
        m_1 = np.sum(self.dataset - mu)
        m_2 = np.sum(np.power((self.dataset - mu), 2))
        z = np.random.normal()
        w = np.random.normal()
        eps = self.eps

        new_mu = mu + eps ** 2 * m_1 / (2 * sigma ** 4) + eps * z
        new_sigma = sigma + 2 * eps ** 2 * m_2 / sigma ** 5 - 2 * N * eps ** 2 / sigma + eps * w

        u = np.random.rand()
        alpha = self.acceptation_rate(mu, sigma, new_mu, new_sigma)
        if u > alpha:
            new_mu = mu
            new_sigma = sigma
        return new_mu, new_sigma, alpha

    def run(self, nb_iter, mu_0, sigma_0):
        mu_estimation = mu_0
        sigma_estimation = sigma_0

        theta_over_time = np.zeros((nb_iter + 1, 2))
        theta_over_time[0] = mu_estimation, sigma_estimation

        alpha_over_time = np.zeros(nb_iter + 1)

        for i in range(nb_iter):
            mu_estimation, sigma_estimation, alpha_over_time[i + 1] = self.iteration(mu_estimation,
                                                                                     sigma_estimation)
            theta_over_time[i + 1] = mu_estimation, sigma_estimation
        return theta_over_time, alpha_over_time
