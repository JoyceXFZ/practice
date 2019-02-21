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
end   = '2014-12-31'
R = pdr.get_data_yahoo("AAPL", start, end)
R = pd.DataFrame(R)['Adj Close'].pct_change()[1:]
R_F = pdr.get_data_yahoo("BIL", start, end)
R_F = pd.DataFrame(R_F)['Adj Close'].pct_change()[1:]
M = pdr.get_data_yahoo("SPY", start, end)
M = pd.DataFrame(M)['Adj Close'].pct_change()[1:]

AAPL_results = regression.linear_model.OLS(R-R_F, sm.add_constant(M)).fit()
aapl_beta = AAPL_results.params[1]

print AAPL_results.summary()
print aapl_beta

prediction = R_F + aapl_beta * (M - R_F)
prediction.plot()
R.plot()
plt.legend(['prediction', 'R'])
plt.show()


