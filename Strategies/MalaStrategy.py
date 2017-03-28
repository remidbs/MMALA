import numpy as np


class MalaStrategy:
    def __init__(self, problem):
        self.problem = problem
        self.d = self.problem.d

    def mu(self, epsilon, x):
        x = x.copy()
        mu = x
        mu += 0.5 * epsilon ** 2 * self.problem.gradient(x)
        return mu

    def sigma(self, epsilon, x):
        sigma = epsilon ** 2 * np.eye(self.d)
        return sigma

    def pi_ratio(self, x, x_new):
        return self.problem.pi_ratio(x, x_new)
