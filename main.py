import numpy as np
import matplotlib.pyplot as plt
import seaborn
from MMALAGaussian import MMALAGaussian
from MALAGaussian import MALAGaussian

import pandas as pd
from pandas.tools.plotting import autocorrelation_plot

'''
	Projet MMALA Simulation on Gaussian 1D framework

	Remi DELBOUYS - Theo LACOMBE

	The goal is to provide a framework to test different manifold prior for gaussian estimation of parameters.
	Compare to standard Langevin approximation.

	Standard items considered are:
	- A dataset of N points drawn from a same 1D gaussian (mu, sigma) in R (sigma^2 > 0)
	- A metric tensor G(mu, sigma)

	To go further :
	- Maybe test with >2D gaussian framework.

	Rem : if we work in dim D, parameters space dimension (for a gaussian) is D + D^2

	Important : code is written to work with Python 3
'''

mu = 0
sigma = 10
N = 1
eps = 0.75
nb_iter = 1000

def big_plot(mu, sigma,N,nb_iter):
    for eps in [0.1,0.75,1.5]:
        mmalag = MMALAGaussian(mu, sigma, N, eps, mu + 1, sigma + 1)
        theta_over_time, alpha_over_time = mmalag.run(nb_iter, 5, 40)
        print(theta_over_time[100:, 0].mean())
        print(theta_over_time[100:, 1].mean())

        plt.figure(1,figsize=(18,10))
        plt.subplot(241)
        plt.plot(theta_over_time[:, 0], theta_over_time[:, 1], '-o')
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.title("MMALA parameter estimation")

        plt.subplot(242)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter + 1)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(243)
        plt.title("Autocorrel mu")
        plt.plot([np.abs(pd.Series(theta_over_time[:,0]).autocorr(i)) for i in range(1,100)])

        plt.subplot(244)
        plt.title("Autocorrel sigma")
        plt.plot([np.abs(pd.Series(theta_over_time[:,1]).autocorr(i)) for i in range(1,100)])

        malag = MALAGaussian(mu, sigma, N, eps, mu + 1, sigma + 1)
        theta_over_time, alpha_over_time = malag.run(nb_iter, 5, 40)

        plt.subplot(245)
        plt.plot(theta_over_time[:, 0], theta_over_time[:, 1], '-o')
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.title("MALA parameter estimation")

        plt.subplot(246)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter + 1)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(247)
        plt.title("Autocorrel mu")
        plt.plot([np.abs(pd.Series(theta_over_time[:,0]).autocorr(i)) for i in range(1,100)])

        plt.subplot(248)
        plt.title("Autocorrel sigma")
        plt.plot([np.abs(pd.Series(theta_over_time[:,1]).autocorr(i)) for i in range(1,100)])

    plt.legend([0.1,0.75,1.5],loc="top right",bbox_to_anchor=(1.5, 1))


big_plot(mu,sigma,N,nb_iter)