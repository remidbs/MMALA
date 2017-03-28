import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Sampler import Sampler
from Strategies.MmalaStrategy import MmalaStrategy
from Strategies.MalaStrategy import MalaStrategy
from Problems.GaussianProblem import GaussianProblem
from Problems.BananaProblem import BananaProblem
from Problems.Banana3DProblem import Banana3DProblem
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def gaussian_samples(nb_iter):
    mu = 0
    sigma = 10
    N = 100

    pb = GaussianProblem(mu, sigma, mu + 1, sigma + 1, N, 10, 40)

    plt.figure(1, figsize=(18, 10))
    for eps in [0.01, 0.75, 10.]:
    # for eps in [0.1, 0.75, 1.5]:
        # MMALA
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

        # MALA
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
    plt.suptitle("Comparison between MMALA and MALA scheme - gaussian shape")
    plt.show()


def banana_samples(nb_iter):
    pb = BananaProblem(100, 0.1, 10, 10)
    plt.figure(2, figsize=(18, 10))
    epsilons = [0.75]  # [1.5, 2.1]
    for eps in epsilons:
        # MMALA
        sampler = Sampler(MmalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()

        plt.subplot(241)
        plt.scatter(theta_over_time[0, :], theta_over_time[1, :])
        plt.xlabel("x1")
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

        # MALA
        sampler = Sampler(MalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()

        plt.subplot(245)
        plt.scatter(theta_over_time[0, :], theta_over_time[1, :])
        plt.xlabel("x1")
        plt.ylabel("x2")
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
    plt.suptitle("Comparison between MMALA and MALA scheme - banana shape")
    plt.show()


def banana3D_samples(nb_iter):
    plt.figure(2, figsize=(18, 10))
    eps = 0.75
    sigmas3 = [100.,1.,0.01]
    for sigma3 in sigmas3:
        pb = Banana3DProblem(100, 0.1, sigma3, [10, 10, 0])

        # MMALA
        sampler = Sampler(MmalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()

        # fig = plt.figure(3)
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(theta_over_time[0, :], theta_over_time[1, :], theta_over_time[2, :])
        # plt.title("MMALA")
        # plt.figure(2)

        plt.subplot(251)
        plt.plot(theta_over_time[0, :], theta_over_time[1, :], '-o')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("MMALA")
        plt.subplot(255)
        plt.plot(theta_over_time[0, :], theta_over_time[2, :], '-o')
        plt.xlabel("x1")
        plt.ylabel("x3")
        plt.title("MMALA")

        plt.subplot(252)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(253)
        plt.title("Autocorrel x1")
        plt.plot([np.abs(pd.Series(theta_over_time[0, :]).autocorr(i)) for i in range(1, 100)])
        plt.ylim(0,1)

        plt.subplot(254)
        plt.title("Autocorrel x2")
        plt.plot([np.abs(pd.Series(theta_over_time[1, :]).autocorr(i)) for i in range(1, 100)])
        plt.ylim(0,1)

        # MALA
        sampler = Sampler(MalaStrategy(pb), epsilon=eps, n_samples=nb_iter)
        theta_over_time, alpha_over_time = sampler.sample()

        # fig = plt.figure(4)
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(theta_over_time[0, :], theta_over_time[1, :], theta_over_time[2, :])
        # plt.title("MALA")
        # plt.figure(2)

        plt.subplot(256)
        plt.plot(theta_over_time[0, :], theta_over_time[1, :], '-o')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("MALA")

        plt.subplot(257)
        plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
        plt.title("Acceptance rate")
        plt.ylim(0, 1)

        plt.subplot(258)
        plt.title("Autocorrel x1")
        plt.plot([np.abs(pd.Series(theta_over_time[0, :]).autocorr(i)) for i in range(1, 100)])
        plt.ylim(0,1)

        plt.subplot(259)
        plt.title("Autocorrel x2")
        plt.plot([np.abs(pd.Series(theta_over_time[1, :]).autocorr(i)) for i in range(1, 100)])
        plt.ylim(0,1)

        plt.subplot(2, 5, 10)
        plt.plot(theta_over_time[0, :], theta_over_time[2, :], '-o')
        plt.xlabel("x1")
        plt.ylabel("x3")
        plt.title("MALA")
    plt.legend(sigmas3, loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.suptitle("Comparison between MMALA and MALA scheme - banana shape")
    plt.show()


# gaussian_samples(3000)
# banana_samples(3000)
banana3D_samples(3000)


algo = "MMALA"
from plot_ellipse import plot_ellipse
problem = BananaProblem(100, 0.1, 10, 10)
if algo=="MALA":
    sp = Sampler(MalaStrategy(problem), epsilon=0.75, n_samples=1000)
else:
    sp = Sampler(MmalaStrategy(problem), epsilon=0.75, n_samples=1000)
x = np.array([10.,10.])
xs = [x]
n_iter = 3200
for i in range(n_iter):
    x_new, x_proposed, _ = sp.one_sample(x.copy())
    if i > (n_iter-200):
        plt.figure()
        plot_ellipse(sp.strategy.mu(0.75,x),sp.strategy.sigma(0.75,x),0.95,"black")
        plt.plot(np.array(xs)[:,0],np.array(xs)[:,1], '-o',markersize=0.1,linewidth=0.3)
        plt.plot([x[0],x_proposed[0]],[x[1],x_proposed[1]], '-o')
        plt.plot([x_proposed[0]],[x_proposed[1]], '-o')
        plt.xlim(-20,20)
        plt.ylim(-25,15)
        plt.title(algo)
        if((x_proposed == x_new).all()):
            plt.legend(["history","current point","proposal (accepted)"],loc="upper right", )
        else:
            plt.legend(["history","current point","proposal (rejected)"],loc="upper right", )
        plt.waitforbuttonpress()
    xs += [x_new]
    x = x_new
