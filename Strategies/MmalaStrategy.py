import numpy as np


class MmalaStrategy:
    def __init__(self, problem):
        self.problem = problem
        self.d = self.problem.d

    def mu(self, epsilon, x):
        Ginv = self.problem.Ginv(x)
        grad_G = self.problem.grad_G(x)
        mu = x \
             + 0.5 * epsilon ** 2 * Ginv.dot(self.problem.gradient(x)) \
             - epsilon ** 2 * np.sum([(Ginv.dot(grad_G[i]).dot(Ginv))[i, :] for i in range(self.d)], axis=1) \
             + 0.5 * epsilon ** 2 * np.sum([Ginv[i, :] * np.trace(Ginv.dot(grad_G[i])) for i in range(self.d)], axis=1)
        return mu

    def sigma(self, epsilon, x):
        Ginv = self.problem.Ginv(x)
        sigma = epsilon ** 2 * Ginv
        return sigma

    def pi_ratio(self, x, x_new):
        return self.problem.pi_ratio(x, x_new)
