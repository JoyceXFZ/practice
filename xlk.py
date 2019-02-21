# XLK etf SPDR
#pip install xlrd  
 
import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import datetime
import numpy as np
import sklearn
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

from sklearn.decomposition import PCA
from numpy import linalg as LA

'''get tech symbol from xlk etf components'''
file = 'C:/python/xlk.xls'
dfs = pd.read_excel(file, sheet_name='Sheet1')
dfs = dfs.fillna('')
dfs.columns = [x.lower() for x in dfs.columns]

sym = []
for i in range(0, len(dfs)-1, 1):
	sym.append(dfs.symbol[i])

'''loading data from yahoo finance'''
start = "2015-01-01"
end = "2018-11-30"
assets = pdr.get_data_yahoo(sym, start, end)
assets = pd.DataFrame(assets)['Close']
print assets.head(10)

'''write the data into excel file'''
#pip install XlsxWriter
#Specify a writer
writer = pd.ExcelWriter('techshares.xlsx', engine = 'xlsxwriter')
#write DataFrame to a file, using the to_excel() function 
assets.to_excel(writer, 'Sheet1')
#save the result
writer.save()	

'''principal components analysis'''
portfolio_returns = assets.pct_change()[1:]
portfolio_returns = portfolio_returns.dropna()
num_pc = 9
X = np.asarray(portfolio_returns)
[n,m] = X.shape
print format(n), format(m)

pca = PCA(n_components=num_pc)
pca.fit(X)

#get the percentage of first three components by PCA
percentage = pca.explained_variance_ratio_
percentage_cum = np.cumsum(percentage)
pca.components = pca.components_
print '{0:.2f}% of the variance is explained by the first 9 PCs'.format(percentage_cum[-1]*100)
