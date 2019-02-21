#normality and heteroskedasticity 

import numpy as np 
import statsmodels.api as sm
from statsmodels import regression, stats
import statsmodels
import matplotlib.pyplot as plt

residuals = np.random.normal(0, 1, 100)

_, pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(residuals)
print pvalue

residuals = np.random.poisson(size = 100)

_, pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(residuals)
print pvalue

xs = np.arange(100)
y1 = xs + 3*np.random.randn(100)

str1 = regression.linear_model.OLS(y1, sm.add_constant(xs)).fit()
fit1 = str1.params[0] + str1.params[1]*xs

plt.scatter(xs, y1)
plt.plot(xs, fit1)
plt.title('Homoskedastic errors')
plt.legend(['predicted', 'observed'])
plt.xlabel('X')
plt.ylabel('Y')

y2 = xs*(1 + 0.5*np.random.randn(100))

str2 = regression.linear_model.OLS(y2, sm.add_constant(xs)).fit()
fit2 = str2.params[0] + str2.params[1]*xs

plt.scatter(xs, y2)
plt.plot(xs, fit2)
plt.title('Heteroskedastic errors')
plt.legend(['predicted', 'observed'])
plt.xlabel('X')
plt.ylabel('Y')

#print str2.summary()

residuals1 = y1-fit1
residuals2 = y2-fit2

xs_with_constant = sm.add_constant(xs)

_,jb_pvalue1, _, _ = statsmodels.stats.stattools.jarque_bera(residuals1)
_,jb_pvalue2, _, _ = statsmodels.stats.stattools.jarque_bera(residuals2)

print 'p value for residuals1 being normal', jb_pvalue1
print 'p value for residuals2 being normal', jb_pvalue2

_, pvalue1, _, _ = stats.diagnostic.het_breushpagan(residuals1, xs_with_constant)
_, pvalue2, _, _ = stats.diagnostic.het_breushpagan(residuals2, xs_with_constant)

print 'p value for residuals1 being heteroskedastic', pvalue1
print 'p value for residuals2 being heteroskedastic', pvalue2

print str1.get_robustcov_results().summary()
print str2.get_robustcov_results().summary()

