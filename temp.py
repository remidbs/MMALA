import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Sampler import Sampler
from Strategies.MmalaStrategy import MmalaStrategy
from Strategies.MalaStrategy import MalaStrategy
from Problems.GaussianProblem import GaussianProblem
import pandas as pd

mu = 0
sigma = 10
N = 1
eps = 0.75
nb_iter = 1000

pb = GaussianProblem(mu, sigma, mu + 1, sigma + 1, N, 10, 40)
sampler = Sampler(MalaStrategy(pb), epsilon=eps, n_samples=nb_iter)

theta_over_time, alpha_over_time = sampler.sample()
print(theta_over_time[100:, 0].mean())
print(theta_over_time[100:, 1].mean())

plt.figure(1, figsize=(18, 10))
plt.subplot(141)
plt.plot(theta_over_time[:, 0], theta_over_time[:, 1], '-o')
plt.xlabel("mu")
plt.ylabel("sigma")
plt.title("MMALA parameter estimation")

plt.subplot(142)
plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter)))
plt.title("Acceptance rate")
plt.ylim(0, 1)

plt.subplot(143)
plt.title("Autocorrel mu")
plt.plot([np.abs(pd.Series(theta_over_time[:, 0]).autocorr(i)) for i in range(1, 100)])

plt.subplot(144)
plt.title("Autocorrel sigma")
plt.plot([np.abs(pd.Series(theta_over_time[:, 1]).autocorr(i)) for i in range(1, 100)])
