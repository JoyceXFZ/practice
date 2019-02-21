import datetime
import numpy as np
import statsmodels.api as sm
from statsmodels import regression, stats
import statsmodels
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

start = '2015-01-01'
end = '2016-01-01'

b1 = pdr.get_data_yahoo('SPY', start, end)
b1 = pd.DataFrame(b1)['Close']
b2 = pdr.get_data_yahoo('JPM', start, end)
b2 = pd.DataFrame(b2)['Close']
 
y = pdr.get_data_yahoo('UNH', start, end)
y = pd.DataFrame(y)['Close']
x = np.arange(len(y))

model = regression.linear_model.OLS(y, sm.add_constant(x)).fit()

prediction = model.params[0] + model.params[1] * x

plt.plot(x, y)
plt.plot(x, prediction, color='r')
plt.legend(['UNH price', 'Regression line'])
plt.xlabel('Time')
plt.ylabel('Price')

plt.show()
print model.summary()

'''
# testing for autocorrelation in both prices and residuals
_, prices_qstats, prices_qstat_pvalues = statsmodels.tsa.stattools.acf(y, qstat=True) 
print 'prices autocorrelation pvalue', prices_qstat_pvalues
_, prices_qstats, prices_qstat_pvalues = statsmodels.tsa.stattools.acf(y-prediction, qstat=True)
print 'residuals autocorrelation pvalue', prices_qstat_pvalues
_,jb_pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(y-prediction)
print 'jarque bera that residuals are normally distributed', jb_pvalue
'''

mlr = regression.linear_model.OLS(y, sm.add_constant(np.column_stack((b1, b2)))).fit()
mlr_prediction = mlr.params[0] + mlr.params[1]*b1 + mlr.params[2]*b2

print 'R squared:', mlr.rsquared_adj
print 't-statistics of coefficient:', mlr.tvalues

y.plot()
mlr_prediction.plot()
plt.show()
plt.legend(['asset', 'model'])
plt.label('price')