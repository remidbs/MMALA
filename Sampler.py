import numpy as np


class Sampler:
    def __init__(self, strategy, epsilon, n_samples):
        self.strategy = strategy
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.d = strategy.problem.d

    def one_sample(self, x):
        mu_x = self.strategy.mu(self.epsilon, x)
        sigma_x = self.strategy.sigma(self.epsilon, x)
        x_proposed = np.random.multivariate_normal(mu_x, sigma_x)
        mu_x_proposed = self.strategy.mu(self.epsilon, x_proposed)
        sigma_x_proposed = self.strategy.sigma(self.epsilon, x_proposed)
        acceptation_rate = min(1, self.strategy.pi_ratio(x, x_proposed) *
                               np.exp(-0.5 * (x - mu_x_proposed).T.dot(np.linalg.inv(sigma_x_proposed)).dot(x - mu_x_proposed) -
                                      -0.5 * (x_proposed - mu_x).T.dot(np.linalg.inv(sigma_x)).dot(x_proposed - mu_x)))
        if (np.random.uniform() < acceptation_rate):
            return x_proposed, acceptation_rate
        else:
            return x, acceptation_rate

    def sample(self):
        samples = np.zeros((self.d, self.n_samples))
        acceptation_rates = np.zeros(self.n_samples)
        samples[:, 0] = self.strategy.problem.starting_point
        acceptation_rates[0] = 1.
        for t in range(1, self.n_samples):
            samples[:, t], acceptation_rates[t] = self.one_sample(samples[:, t - 1])
        return samples, acceptation_rates
