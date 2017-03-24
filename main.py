import numpy as np
import matplotlib.pyplot as plt
import seaborn
from MMALAGaussian import MMALAGaussian
from MALAGaussian import MALAGaussian

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
N = 30
eps = 0.75
nb_iter = 200

mmalag = MMALAGaussian(mu,sigma,N,eps,1,11)
theta_over_time, alpha_over_time = mmalag.run(nb_iter, 5, 40)

plt.plot(theta_over_time[:, 0], theta_over_time[:, 1], '-o')
plt.xlabel("mu estimation over time")
plt.ylabel("sigma estimation over time")
plt.title("Subsequent iteration of MMALA algorithm for parameter esimation")

plt.figure(2)
plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter + 1)))
plt.show()

malag = MALAGaussian(mu,sigma,N,eps,1,11)
theta_over_time, alpha_over_time = malag.run(nb_iter, 5, 40)

plt.plot(theta_over_time[:, 0], theta_over_time[:, 1], '-o')
plt.xlabel("mu estimation over time")
plt.ylabel("sigma estimation over time")
plt.title("Subsequent iteration of MALA algorithm for parameter esimation")

plt.figure(2)
plt.plot(alpha_over_time.cumsum() / np.cumsum(np.ones(nb_iter + 1)))
plt.show()