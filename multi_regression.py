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


#real finance data testing
start = '2017-01-01'
end = '2017-12-31'	
asset1 = pdr.get_data_yahoo('ISRG', start, end)
asset1 = pd.DataFrame(asset1)['Open']
asset2 = pdr.get_data_yahoo('BMW', start, end)
asset2 = pd.DataFrame(asset1)['Open']
benchmark = pdr.get_data_yahoo('SPY', start, end)
benchmark = pd.DataFrame(benchmark)['Open']

slr = regression.linear_model.OLS(asset1, sm.add_constant(asset2)).fit()
print slr.params[1]


mlr = regression.linear_model.OLS(asset1, sm.add_constant(np.column_stack((asset2, benchmark)))).fit()
prediction = mlr.params[0] + mlr.params[1]*asset2 + mlr.params[2]*benchmark
prediction.name = 'prediction'

prediction.plot(color='y')
plt.xlabel('price')
plt.legend()
plt.show()

print mlr.summary()