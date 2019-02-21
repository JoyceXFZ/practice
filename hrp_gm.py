'''
last paragraph is not finished successfully
'''

#hierarchical risk parity 
import numpy as np
from scipy.linalg import block_diag
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override()


#symbols = ['EEM', 'EWG', 'TIP']

symbols = ['EEM', 'EWG', 'TIP', 'EWJ', 'EFA', 'IEF', 'EWQ', 
           'EWU', 'XLB', 'XLE', 'XLF', 'LQD', 'XLK', 'XLU', 
           'EPP', 'FXI', 'VGK', 'VPL', 'SPY', 'TLT', 'BND', 
           'CSJ', 'DIA']

start = '2018-01-01'
end = '2018-12-31'

asset = pdr.get_data_yahoo(symbols, start, end)
asset = pd.DataFrame(asset)['Close']
rets = asset.pct_change()
eoms = rets.resample('1BM').mean()[1:]

estimate_correl = eoms.corr(method='pearson')
estimate_covar = eoms.cov()

plt.pcolormesh(estimate_correl)
plt.colorbar()
plt.title('Estimated correlation matrix')
plt.show()

plt.pcolormesh(estimate_covar)
plt.colorbar()
plt.title('Estimated covariance matrix')
plt.show()

distances = np.sqrt((1 - estimate_correl) / 2)

#
def seriation(Z, N, cur_index):
    """Returns the order implied by a hierarchical tree (dendrogram).
    
       :param Z: A hierarchical tree (dendrogram).
       :param N: The number of points given to the clustering process.
       :param cur_index: The position in the tree for the recursive traversal.
       
       :return: The order implied by the hierarchical tree Z.
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))

    
def compute_serial_matrix(dist_mat, method="ward"):
    """Returns a sorted distance matrix.
    
       :param dist_mat: A distance matrix.
       :param method: A string in ["ward", "single", "average", "complete"].
        
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a,b = np.triu_indices(N, k=1)
    seriated_dist[a,b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(distances.values, method='single')

plt.pcolormesh(distances)
plt.colorbar()
plt.title('Original order distance matrix')
plt.show()

plt.pcolormesh(ordered_dist_mat)
plt.colorbar()
plt.title('Re-ordered distance matrix')
plt.show()


def compute_HRP_weights(covariances, res_order):
	weights = pd.Series(1, index=res_order)
	clustered_alphas = [res_order]
	
	while len(clustered_alphas) > 0:
		clustered_alphas = [cluster[start:end] for cluster in clustered_alphas for start, end in ( (0, len(cluster) // 2),(len(cluster) // 2, len(cluster))) if len(cluster) > 1]
		
		for subcluster in range(0, len(clustered_alphas), 2):
			clustered_alphas = pd.DataFrame(clustered_alphas)

			left_cluster = clustered_alphas[subcluster]
			right_cluster = clustered_alphas[subcluster + 1]
			left_subcovar = covariances[left_cluster].iloc[left_cluster]
			inv_diag = 1 / np.diag(left_subcovar.values)
			parity_w = inv_diag * (1 / np.sum(inv_diag))
			left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))
			
			right_subcovar = covariances[right_cluster].iloc[right_cluster]
			inv_diag = 1 / np.diag(right_subcovar.values)
			parity_w = inv_diag * (1 / np.sum(inv_diag))
			right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))
			
			alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)
			
			weights[left_cluster] *= alloc_factor
			weights[right_cluster] *= 1 - alloc_factor
	
	return weights

HRP_weights = compute_HRP_weights(estimate_covar, res_order)
#print(round((HRP_weights * eoms).sum(axis=1).std() * np.sqrt(252),2))	

'''

def compute_MV_weights(covariances):
    inv_covar = np.linalg.inv(covariances)
    u = np.ones(len(covariances))
    
    return np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))


def compute_RP_weights(covariances):
    weights = (1 / np.diag(covariances)) 
    
    return weights / sum(weights)


def compute_unif_weights(covariances):
    
    return [1 / len(covariances) for i in range(len(covariances))]
	

unif_weights = compute_unif_weights(estimate_covar)

print(round((unif_weights * eoms).sum(axis=1).std() * np.sqrt(252),
            2))
			
RP_weights = compute_RP_weights(estimate_covar)

print(round((RP_weights * eoms).sum(axis=1).std() * np.sqrt(252),
      2))
	  
MV_weights = compute_MV_weights(estimate_covar)

print(round((MV_weights * eoms).sum(axis=1).std() * np.sqrt(252),
            2))
MV_weights = compute_MV_weights(estimate_covar)

print(round((MV_weights * eoms).sum(axis=1).std() * np.sqrt(252),
            2))
'''