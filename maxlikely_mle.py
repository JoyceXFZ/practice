#maximum likehood estimation
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy 
import scipy.stats

#exponential distribution
lamda = 5
X = np.random.exponential(lamda, 1000)

def exp_lamda_MLE(X):
	T = len(X)
	s = sum(X)
	return s/T

pdf = scipy.stats.expon.pdf
x = range(0, 80)
plt.hist(X, bins=x, normed='true')
plt.plot(pdf(x, scale=1))
plt.xlabel('value')
plt.ylabel('observed frequency')
plt.legend(['Fitted distribution pdf', 'Observed data'])
plt.show()


'''
#normal distribution

mean = 40
std = 10
X = np.random.normal(mean, std, 1000)

def normal_mu_MLE(X):
	T = len(X)
	s = sum(X)
	return 1.0/T * s

	
def normal_sigma_MLE(X):
	T = len(X)
	mu = normal_mu_MLE(X)
	s = sum(np.power((X-mu), 2))
	sigma_squared = 1.0/T * s
	return math.sqrt(sigma_squared)


mu, std = scipy.stats.norm.fit(X)
pdf = scipy.stats.norm.pdf
x = np.linspace(0, 80, 80)
plt.hist(X, bins=x, normed='true')
plt.plot(pdf(x, loc=mu, scale=std))
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend(['fitted distribution pdf', 'observed data'])
plt.show()
'''