import numpy as np 
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
import math 

import datetime
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

start = '2014-01-01'
end   = '2015-01-01'
asset = pdr.get_data_yahoo("TSLA", start, end)
asset = pd.DataFrame(asset)['Adj Close']
benchmark = pdr.get_data_yahoo("SPY", start, end)
benchmark = pd.DataFrame(benchmark)['Adj Close']

r_a = asset.pct_change()[1:]
r_b = benchmark.pct_change()[1:]

X = r_b.values
Y = r_a.values

def linreg(x, y):
	x = sm.add_constant(x)
	model = regression.linear_model.OLS(y,x).fit()
	x = x[:,1]
	return model.params[0], model.params[1]
	
alpha, beta = linreg(Y, X)
print alpha, beta
portfolio = -beta * r_b + r_a
portfolio.name = 'TSLA+hedge'

r_a.plot()
r_b.plot()
portfolio.plot()
plt.legend(['A', 'B', 'hedge'])
plt.show()
print portfolio.mean(), r_a.mean()
print portfolio.std(), r_a.std()