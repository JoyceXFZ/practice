#linear regression
#pip install fix_yahoo_finance

from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import datetime
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

import seaborn
import scipy 
import scipy.stats

def linreg(X,Y):
	X = sm.add_constant(X)
	model = regression.linear_model.OLS(Y,X).fit()
	a = model.params[0]
	b = model.params[1]
	X = X[:,1]
	
	X2 = np.linspace(X.min(), X.max(), 100)
	Y_hat = X2* b + a
	plt.scatter(X, Y, alpha=0.3)
	plt.plot(X2, Y_hat, 'r', alpha=0.9)
	plt.xlabel('X value')
	plt.ylabel('Y value')
	plt.show()
	print model.summary()
	return model.summary()

#real finance data testing


start = '2014-01-01'
end = '2015-01-01'	
asset = pdr.get_data_yahoo('TSLA', start, end)
asset = pd.DataFrame(asset)['Close']
benchmark = pdr.get_data_yahoo('SPY', start, end)
benchmark = pd.DataFrame(benchmark)['Close']

# linear regression
r_a = asset.pct_change()[1:]
r_b = benchmark.pct_change()[1:]
linreg(r_b.values, r_a.values) 

#MLE for asset returns 
abs_returns = np.diff(asset)
returns = abs_returns/asset[:-1]
mu, std = scipy.stats.norm.fit(returns)
pdf = scipy.stats.norm.pdf
x = np.linspace(-1, 1, num=100)
h = plt.hist(returns, bins=x, normed='true')
plt.plot(x, pdf(x, loc=mu, scale=std))
plt.show()

#seaborn.regplot(r_b.values, r_a.values)
'''
X = np.random.rand(100)
Y = X + 0.2 * np.random.rand(100)
linreg(X, Y)'''