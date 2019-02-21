import datetime
import numpy as np
import statsmodels.api as sm
from statsmodels import regression, stats
import statsmodels
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import math
import statsmodels.stats.diagnostic as smd

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

start = '2017-01-01'
end = '2019-01-01'

asset = pdr.get_data_yahoo('AAPL', start, end)
asset = pd.DataFrame(asset)['Close']

benchmark1 = pdr.get_data_yahoo('XLK', start, end)
benchmark1 = pd.DataFrame(benchmark1)['Close']
benchmark2 = pdr.get_data_yahoo('QQQ', start, end)
benchmark2 = pd.DataFrame(benchmark2)['Close']
print asset.head(), benchmark1.head(), benchmark2.head()
fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.plot(asset,c = 'b', label = 'AAPL, left')
ax.plot(benchmark2, c = 'k', label = 'QQQ, left')
ax.set_ylim([110, 250])
ax.legend(loc=2)
ax2 = ax.twinx()
ax2.plot(benchmark1, c = 'm', label = 'XLK, right')
ax2.set_ylim([45, 85])
ax2.legend()

plt.title('Chart of Apple Prices and Tech ETF Indexes')
plt.show() 

# log difference is more or less the same as daily returns
#r_a = np.log(asset)
#r_b = np.log(benchmark)
#r_a = np.diff(r_a[1:])
#r_b = np.diff(r_b[1:])

benchmark = pdr.get_data_yahoo('XLK', start, end)
benchmark = pd.DataFrame(benchmark)['Close']

r_a = asset.pct_change()[1:].values
r_b = benchmark.pct_change()[1:].values
 
r_b = sm.add_constant(r_b)
model = sm.OLS(r_a, r_b).fit()

r_b = r_b[:, 1]
B0, B1 = model.params

A_hat = (B1 * r_b + B0)
plt.scatter(r_b, r_a, alpha = 1)
plt.plot(r_b, A_hat, 'r', alpha = 1)
plt.title('Return')
plt.xlabel('Apple returns')
plt.ylabel('XLK returns')
plt.show()
print 'estimated apple beta: ', B1
print model.summary()

residuals = model.resid
plt.scatter(model.predict(), residuals)
plt.axhline(0, color = 'red')
plt.xlabel('Apple returns')
plt.ylabel('Residuals')
plt.title('Residuals')
plt.show()

#breusch-pagan heteroscedasticity test for residuals
bp_test = smd.het_breuschpagan(residuals, model.model.exog)
print 'lagrange multiplier statistics: ', bp_test[0]
print 'p-value: ', bp_test[1]
print 'f-value: ', bp_test[2]
print 'f-p-value: ', bp_test[3], '\n'
if bp_test[1] > 0.05:
	print 'the relationship is no heteroscedasticity'
if bp_test[1] < 0.05:
	print 'the relationship is heteroscedasticity'

#testing residuals autocorrelation
ljung_box = smd.acorr_ljungbox(r_a)
print 'p-value: ', ljung_box[1], '\n'
if any(ljung_box[1] < 0.05):
	print 'the residuals are autocorrelated'
else:
	print 'the residuals are not autocorrelated'

