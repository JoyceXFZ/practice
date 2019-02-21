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

from scipy import poly1d

#real finance data testing 
start = '2015-01-01'
end = '2016-01-01'	
x1 = pdr.get_data_yahoo('CELG', start, end)
x1 = pd.DataFrame(x1)['Adj Close']
x2 = pdr.get_data_yahoo('MCD', start, end)
x2 = pd.DataFrame(x2)['Adj Close']
x3 = pdr.get_data_yahoo('UNH', start, end)
x3 = pd.DataFrame(x3)['Adj Close']
x4 = pdr.get_data_yahoo('SPY', start, end)
x4 = pd.DataFrame(x4)['Adj Close']
y = pdr.get_data_yahoo('ISRG', start, end)
y = pd.DataFrame(y)['Adj Close']

#trade using a simple mean-reversion strategy
def trade(stock, length):
	if length == 0:
		return 0
	
	rolling_window = stock.rolling(window=length)
	mu  = rolling_window.mean()
	std = rolling_window.std()
	
	zscores = (stock - mu)/std
	
	money = 0
	count = 0
	for i in range(len(stock)):
		if zscores[i] > 1:
			money += stock[i]
			count -= 1
		elif zscores[i] < -1:
			money -= stock[i]
			count += 1
		elif abs(zscores[i]) < 0.5:
			money += count*stock[i]
			count = 0
	return money
	
length_scores = [trade(y, l) for l in range(255)]
best_length = np.argmax(length_scores)
print 'best window length: ', best_length


start2 = '2016-01-01'
end2 = '2017-01-01'	
y2 = pdr.get_data_yahoo('ISRG', start2, end2)
y2 = pd.DataFrame(y2)['Adj Close']

length_scores2 = [trade(y2, l) for l in range(255)]
print best_length, 'day window: ', length_scores2[best_length]

best_length2 = np.argmax(length_scores2)
print best_length2, 'day window:', length_scores2[best_length2]

plt.plot(length_scores)
plt.plot(length_scores2)
plt.xlabel('window length')
plt.ylabel('score')
plt.legend(['2015-2016', '2016-2017'])
plt.show()


'''
#rolling windows
mu_30d = x1.rolling(window=30).mean()
mu_60d = x1.rolling(window=60).mean()
mu_100d = x1.rolling(window=100).mean()

plt.plot(x1[100:], label='Asset')
plt.plot(mu_30d[100:], label='30d MA')
plt.plot(mu_60d[100:], label='60d MA')
plt.plot(mu_100d[100:], label='100d MA')
plt.xlabel('day')
plt.ylabel('price')
plt.legend()
plt.show()
'''

'''
#in-sample
slr = regression.linear_model.OLS(y, sm.add_constant(x1)).fit()
slr_prediction = slr.params[0] + slr.params[1]*x1

mlr = regression.linear_model.OLS(y, sm.add_constant(np.column_stack((x1, x2, x3, x4)))).fit()
mlr_prediction = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4

print slr.rsquared_adj
print slr.f_pvalue
print mlr.rsquared_adj
print mlr.f_pvalue
'''
##################################################
'''
#out-of-sample
start = '2016-01-01'
end = '2017-01-01'	
x1 = pdr.get_data_yahoo('CELG', start, end)
x1 = pd.DataFrame(x1)['Adj Close']
x2 = pdr.get_data_yahoo('MCD', start, end)
x2 = pd.DataFrame(x2)['Adj Close']
x3 = pdr.get_data_yahoo('UNH', start, end)
x3 = pd.DataFrame(x3)['Adj Close']
x4 = pdr.get_data_yahoo('SPY', start, end)
x4 = pd.DataFrame(x4)['Adj Close']
y = pdr.get_data_yahoo('ISRG', start, end)
y = pd.DataFrame(y)['Adj Close']

slr_prediction2 = slr.params[0] + slr.params[1]*x1
mlr_prediction2 = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4

#adj for slr model 
p = 1
N = len(y)
adj1 = float(N-1)/(N-p-1)

#adj for mlr
p = 4
N = len(y)
adj2 = float(N-1)/(N-p-1)

SST = sum((y - np.mean(y))**2)
SSRs = sum((slr_prediction2 - y)**2)
print 1-adj1*SSRs/SST
SSRm = sum((mlr_prediction2 - y)**2)
print 1-adj2*SSRm/SST

y.plot()
slr_prediction2.plot()
mlr_prediction2.plot()
plt.legend(['ISRG', 'SLR', 'MLR'])
plt.show()
'''