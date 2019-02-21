#GARCH model 

import cvxopt
from functools import partial
import math
import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt

a0 = 1.0
a1 = 0.1
b1 = 0.8
sigma1 = math.sqrt(a0 / (1- a1 - b1))

def simulate_GARCH(T, a0, a1, b1, sigma1):
	X = np.ndarray(T)
	sigma = np.ndarray(T)
	sigma[0] = sigma1
	
	for t in range(1, T):
		X[t-1] = sigma[t-1] * np.random.normal(0, 1)
		sigma[t] = math.sqrt(a0 + b1 * sigma[t-1]**2 + a1 * X[t-1]**2 )
	
	X[T-1] = sigma[T-1] * np.random.normal(0,1)
	
	return X, sigma

X, _ = simulate_GARCH(10000, a0, a1, b1, sigma1)
X = X[1000:]
X = X / np.std(X)

def compile_tails_to_normal(X):
	A = np.zeros((2, 4))
	
	for k in range(4):
		A[0, k] = len(X[X > (k+1)]) / float(len(X))
		A[1, k] = 1 - stats.norm.cdf(k+1)
		
		return A

print compile_tails_to_normal(X)
plt.hist(X, bins=50)
plt.xlabel('sigma')
plt.ylabel('observations')
plt.show()

X2 = np.random.normal(0, 1, 9000)
both = np.matrix([X, X2])

plt.plot(both.T, alpha=0.7)
plt.axhline(X2.std(), color='yellow', linestyle= '--')
plt.axhline(-X2.std(), color='yellow', linestyle= '--')
plt.axhline(3*X2.std(), color='red', linestyle= '--')
plt.axhline(-3*X2.std(), color='red', linestyle= '--')
plt.xlabel('time')
plt.ylabel('sigma')
plt.show()


''' testing for ARCH'''
X, _ = simulate_GARCH(1100, a0, a1, b1, sigma1)
X = X[100:]

p = 20

Y2 = (X**2)[p:]
X2 = np.ndarray((980, p))
for i in range(p, 1000):
	X2[i-p, :] = np.asarray((X**2)[i-p:i])[::-1]
	
model = sm.OLS(Y2, X2)
model = model.fit()
print model.summary()
theta = np.matrix(model.params)
omega = np.matrix(model.cov_HC0)
F = np.asscalar(theta * np.linalg.inv(omega) * theta.T)

print np.asarray(theta.T).shape

plt.plot(range(20), np.asarray(theta.T))
plt.xlabel('Lag Amount')
plt.ylabel('Estimated Coefficient for Lagged Dadapoint')
plt.show()

print 'F=' + str(F)

chi2dist = scipy.stats.chi2(p)
pvalue = 1 - chi2dist.cdf(F)
print 'p-value = ' + str(pvalue)

print theta/np.diag(omega)

'''fitting GARCH with MLE'''
X, _ = simulate_GARCH(10000, a0, a1, b1, sigma1)
X = X[1000:]

def compute_squared_sigmas(X, initial_sigma, theta):
	a0 = theta[0]
	a1 = theta[1]
	b1 = theta[2]
	
	T = len(X)
	sigma2 = np.ndarray(T)
	
	sigma2[0] = initial_sigma **2
	
	for t in range(1, T):
		sigma2[t] = a0 + a1 *X[t-1]**2 + b1 * sigma2[t-1]
	
	return sigma2

	
plt.plot(range(len(X)), compute_squared_sigmas(X, np.sqrt(np.mean(X**2)), (1, 0.5, 0.5)))
plt.xlabel('Time')
plt.ylabel('Sigma')
plt.show()


def negative_log_likelihood(X, theta):
	T = len(X)
	
	initial_sigma = np.sqrt(np.mean(X**2))
	sigma2 = compute_squared_sigmas(X, initial_sigma, theta)
	
	return -sum([-np.log(np.sqrt(2.0 * np.pi)) -
	(X[t]**2) / (2.0 * sigma2[t]) - 0.5 * np.log(sigma2[t]) 
	for t in range(T)])
	

objective = partial(negative_log_likelihood, X)

def constraint1(theta):
	return np.array([1 - (theta[1] + theta[2])])

def constraint2(theta):
	return np.array([theta[1]])
	
def constraint3(theta):
	return np.array([theta[2]])

cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3})

		
result = scipy.optimize.minimize(objective, (1, 0.5, 0.5),
                                 method='SLSQP',
								 constraints=cons)

theta_mle = result.x
print 'theta MLE: ' + str(theta_mle)

def check_theta_estimate(X, theta_estimate):
	initial_sigma = np.sqrt(np.mean(X**2))
	sigma = np.sqrt(compute_squared_sigmas(X, initial_sigma, theta_estimate))
	epsilon = X / sigma
	print 'Tail table'
	print compile_tails_to_normal(epsilon / np.std(epsilon))
	print ''
	_, pvalue, _, _ = jarque_bera(epsilon)
	print 'jarque_bera probability normal: ' + str(pvalue)

check_theta_estimate(X, theta_mle)

''' General methods of moments for estimating GARCH parameters'''
def standardized_moment(x, mu, sigma, n):
	return ((x - mu) ** n) / (sigma ** n)

def gmm_objective(X, W, theta):
	initial_sigma = np.sqrt(np.mean(X**2))
	sigma = np.sqrt(compute_squared_sigmas(X, initial_sigma, theta))
	e = X / sigma
	
	m1 = np.mean(e)
	m2 = np.mean(e**2) -1
	m3 = np.mean(standardized_moment(e, np.mean(e), np.std(e), 3))
	m4 = np.mean(standardized_moment(e, np.mean(e), np.std(e), 4) -3)
	
	G = np.matrix([m1, m2, m3, m4]).T 
	
	return np.asscalar(G.T * W * G)

def gmm_variance(X, theta):
	initial_sigma = np.sqrt(np.mean(X**2))
	sigma = np.sqrt(compute_squared_sigmas(X, initial_sigma, theta))
	e = X / sigma
	
	m1 = e**2
	m2 = (e**2 -1) ** 2
	m3 = standardized_moment(e, np.mean(e), np.std(e), 3)**2
	m4 = (standardized_moment(e, np.mean(e), np.std(e), 4)-3)**2
	
	T = len(X)
	s = np.ndarray((4,1))
	for t in range(T):
		G = np.matrix([m1[t], m2[t], m3[t], m4[t]]).T
		s = s + G * G.T
		
	return s / T
	
W = np.identity(4)
gmm_iterations = 10

theta_gmm_estimate = theta_mle
for i in range(gmm_iterations):
	objective = partial(gmm_objective, X, W)
	result = scipy.optimize.minimize(objective, theta_gmm_estimate, constraints=cons)
	theta_gmm_estimate = result.x 
	print 'iteration' + str(i) + 'theta: ' + str(theta_gmm_estimate)

	W = np.linalg.inv(gmm_variance(X, theta_gmm_estimate))

check_theta_estimate(X, theta_gmm_estimate)

'''predict the future'''
sigma_hats = np.sqrt(compute_squared_sigmas(X, np.sqrt(np.mean(X**2)), theta_mle))
initial_sigma = sigma_hats[-1]
print initial_sigma

a0_estimate = theta_gmm_estimate[0]
a1_estimate = theta_gmm_estimate[1]
b1_estimate = theta_gmm_estimate[2]

X_forecast, sigma_forecast = simulate_GARCH(100, a0_estimate, a1_estimate, b1_estimate, initial_sigma)
plt.plot(range(-100, 0), X[-100:], 'b-')
plt.plot(range(-100, 0), sigma_hats[-100:], 'r-')
plt.plot(range(0, 100), X_forecast[-100:], 'b--')
plt.plot(range(0, 100), sigma_forecast[-100:], 'r--')
plt.xlabel('Time')
plt.legend(['X', 'sigma'])
plt.show()


plt.plot(range(-100, 0), X[-100:], 'b-')
plt.plot(range(-100, 0), sigma_hats[-100:], 'r-')
plt.xlabel('Time')
plt.legend(['X', 'sigma'])

max_X = [-np.inf]
min_X = [np.inf]
for i in range(100):
	X_forecast, sigma_forecast = simulate_GARCH(100, a0_estimate, a1_estimate, b1_estimate, initial_sigma)
	if max(X_forecast) > max(max_X):
		max_X = X_forecast
	elif min(X_forecast) < min(max_X):
		min_X = X_forecast
	plt.plot(range(0, 100), X_forecast, 'b--', alpha=0.05)
	plt.plot(range(0, 100), sigma_forecast, 'r--', alpha=0.05)

plt.plot(range(0, 100), max_X, 'y--', alpha=1)
plt.plot(range(0, 100), min_X, 'y--', alpha=1)
plt.show()
