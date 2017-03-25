import numpy as np


class BananaProblem:
    def __init__(self, a, B, starting_mu, starting_sigma):
        self.starting_point = np.array([starting_mu, starting_sigma])
        self.d = 2
        self.a = a * 1.
        self.B = B * 1.

    def pi_ratio(self, x, x_new):
        return np.exp(-0.5 * x_new[0] ** 2 / self.a - 0.5 * (x_new[1] + self.B * x_new[0] ** 2 - self.a * self.B) ** 2 \
                      + 0.5 * x[0] ** 2 / self.a + 0.5 * (x[1] + self.B * x[0] ** 2 - self.a * self.B) ** 2)

    def gradient(self, x):
        g1 = -x[0] / self.a - 2 * self.B * x[0] * (x[1] + self.B * x[0] ** 2 - self.a * self.B)
        g2 = -x[1] - self.B * x[0] ** 2 + self.a * self.B
        return np.array([g1, g2])

    def Ginv(self, x):
        G = -np.array(
            [[-1 / self.a - 2 * self.B * x[1] - 6 * self.B ** 2 * x[0] ** 2 - 2 * self.a * self.B ** 2,
              -2 * self.B * x[0]],
             [-2 * self.B * x[0], -1.]])
        Ginv = np.linalg.inv(G)
        return Ginv

    def grad_G(self, x):  # renvoie la liste des derivees de G en la ieme variable
        return [np.array([[-12 * self.B ** 2 * x[0], -2 * self.B], [-2 * self.B, 0]]),
                np.array([[-2 * self.B, 0], [0, 0]])]
