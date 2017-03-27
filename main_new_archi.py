import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Sampler import Sampler
from Strategies.MmalaStrategy import MmalaStrategy
from Strategies.MalaStrategy import MalaStrategy
from Problems.GaussianProblem import GaussianProblem
from Problems.BananaProblem import BananaProblem
import pandas as pd



def gaussian_samples(nb_iter):
    mu = 0
    sigma = 10
    N = 100

    pb = GaussianProblem(mu, sigma, mu + 1, sigma + 1, N, 10, 40)

    plt.figure(1, figsize=(18, 10))
    for eps in [0.1, 0.75, 1.5]:
        #MMALA
        sampler = Sampler(MmalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()


        plt.subplot(241)
        plt.plot(theta_over_time[0, :], theta_over_time[1, :], '-o')
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.title("MMALA parameter estimation")

        plt.subplot(242)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(243)
        plt.title("Autocorrel mu")
        plt.plot([np.abs(pd.Series(theta_over_time[0, :]).autocorr(i)) for i in range(1, 100)])

        plt.subplot(244)
        plt.title("Autocorrel sigma")
        plt.plot([np.abs(pd.Series(theta_over_time[1, :]).autocorr(i)) for i in range(1, 100)])

        #MALA
        sampler = Sampler(MalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()

        plt.subplot(245)
        plt.plot(theta_over_time[0, :], theta_over_time[1, :], '-o')
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.title("MALA parameter estimation")

        plt.subplot(246)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(247)
        plt.title("Autocorrel mu")
        plt.plot([np.abs(pd.Series(theta_over_time[0, :]).autocorr(i)) for i in range(1, 100)])

        plt.subplot(248)
        plt.title("Autocorrel sigma")
        plt.plot([np.abs(pd.Series(theta_over_time[1, :]).autocorr(i)) for i in range(1, 100)])

    plt.legend([0.1, 0.75, 1.5], loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.show()



def banana_samples(nb_iter):
    pb = BananaProblem(100,0.1,10,10)
    plt.figure(2, figsize=(18, 10))
    epsilons = [0.5,1.0,1.5]
    for eps in epsilons:
        #MMALA
        sampler = Sampler(MmalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()


        plt.subplot(241)
        plt.plot(theta_over_time[0, :], theta_over_time[1, :], '-o')
        plt.xlabel("x2")
        plt.ylabel("x2")
        plt.title("MMALA")

        plt.subplot(242)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(243)
        plt.title("Autocorrel x1")
        plt.plot([np.abs(pd.Series(theta_over_time[0, :]).autocorr(i)) for i in range(1, 100)])

        plt.subplot(244)
        plt.title("Autocorrel x2")
        plt.plot([np.abs(pd.Series(theta_over_time[1, :]).autocorr(i)) for i in range(1, 100)])

        #MALA
        sampler = Sampler(MalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()

        plt.subplot(245)
        plt.plot(theta_over_time[0, :], theta_over_time[1, :], '-o')
        plt.xlabel("mu")
        plt.ylabel("sigma")
        plt.title("MALA")

        plt.subplot(246)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(247)
        plt.title("Autocorrel x1")
        plt.plot([np.abs(pd.Series(theta_over_time[0, :]).autocorr(i)) for i in range(1, 100)])

        plt.subplot(248)
        plt.title("Autocorrel x2")
        plt.plot([np.abs(pd.Series(theta_over_time[1, :]).autocorr(i)) for i in range(1, 100)])

    plt.legend(epsilons, loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.show()

banana_samples(1000)