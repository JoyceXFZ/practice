import datetime
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

symbol = ['JPM', 'MSFT', 'FB', 'AAPL', 'MCD', 'ABX', 'NEM', 'BA', 'AEM', 'UNH']
start = '2015-01-01'
end = '2018-11-30'

prices = pdr.get_data_yahoo(symbol, start, end)
portfolio_returns = pd.DataFrame(prices)['Close'].pct_change()[1:]

num_pc = 3
X = np.asarray(portfolio_returns)
[n,m] = X.shape
print format(n), format(m)

pca = PCA(n_components=num_pc)
pca.fit(X)

percentage = pca.explained_variance_ratio_
percentage_cum = np.cumsum(percentage)
pca.components = pca.components_
print '{0:.2f}% of the variance is explained by the first 3 PCs'.format(percentage_cum[-1]*100)

x = np.arange(1, len(percentage)+1, 1)
plt.subplot(1,2,1)
plt.bar(x, percentage*100)
plt.title('Contribution of principal components', fontsize= 16)
plt.xlabel('principal components', fontsize = 16)
plt.ylabel('percentage', fontsize = 16)
plt.xticks(x, fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim([0, num_pc+1])

plt.subplot(1,2,2)
plt.plot(x, percentage_cum*100)
plt.xlabel('principal components', fontsize = 16)
plt.ylabel('percentage', fontsize = 16)
plt.title('Cumulative contribution of principal components', fontsize= 16)
plt.xticks(x, fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim([0, num_pc])
plt.ylim([50, 80])
plt.show()


factor_returns = X.dot(pca.components.T)
factor_returns = pd.DataFrame(columns=['factor1', 'factor2', 'factor3'],
                              index = portfolio_returns.index,
							  data = factor_returns)
print factor_returns.head()


factor_exposures = pd.DataFrame(index=['factor1', 'factor2', 'factor3'],
                                columns=portfolio_returns.columns,
								data = pca.components_).T
print factor_exposures.head()


labels = factor_exposures.index
data = factor_exposures.values
plt.subplots_adjust(bottom = 0.1)
plt.scatter(
	data[:,0], data[:, 1], marker='o', s=300, c='m',
	cmap=plt.get_cmap('Spectral'))
plt.title('Coefficients of PC1 and PC2')
plt.xlabel('factor exposure of PC1')
plt.ylabel('factor exposure of PC2')
plt.xlim([-0.8, 0.2])


for label, x, y in zip(labels, data[:,0], data[:,1]):
	plt.annotate(
	label,
	xy=(x,y), xytext=(-20, 20),
	textcoords='offset points', ha='right', va='bottom',
	bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.5),
	arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
	)
	
plt.show()
