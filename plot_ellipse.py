# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
	Plots the conic of equation xT.A.x + B.x + C = 0
'''


def plot_conic(A, B, C, color):
    range_u = np.linspace(-20, 20, 400)
    range_v = np.linspace(-30, 20, 400)
    u, v = np.meshgrid(range_u, range_v)

    z = A[0, 0] * u ** 2 + (A[0, 1] + A[1, 0]) * u * v + A[1, 1] * v ** 2
    z += B[0] * u + B[1] * v
    z += C

    plt.contour(u, v, z, [0], colors=color)


'''
    Plots the ellipse that contains p percent of the gaussian distribution
    of average mu, and covariance matrix sigma.
    
'''


def plot_ellipse(mu, sigma, p, color):
    A = np.linalg.inv(sigma)
    B = -2 * np.dot(A, mu)
    C = np.dot(mu, np.dot(A, mu)) + 2 * np.log(1 - p)

    plot_conic(A, B, C, color)
