# principle component analysis
from numpy import linalg as LA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_test_image(m, n):
	X = np.zeros((m, n))
	X[25:80, 25:80] = 1
	for i in range(25, 80, 1):
		X[i+80:160, 100+i-1] = 2
	for i in range(0, 200, 1):
		for j in range(0, 200, 1):
			if ((i-135)**2 + (j-53)**2) <= 900:
				X[i, j] = 3
	return X
	
X = generate_test_image(200, 200)

implot = plt.imshow(X, cmap = 'gray')
plt.title('Original Test Image of PCA')	
plt.show()

m = X.shape[0]
n = X.shape[1]
X = np.asarray(X, dtype=np.float64)
C = np.cov(X)
np.linalg.matrix_rank(C)

P, L = LA.eigh(C)
P = P[::-1]
L = L[:, ::-1]

'''
print np.allclose(L.dot(np.diag(P).dot(L.T)), C)
plt.semilogy(P, '-o')
plt.xlim([1, P.shape[0]])
plt.xlabel('eigenvalue index')
plt.ylabel('eigenvalue in a log scale')
plt.title('Eigenvalues of Covariance Matrix')
'''
 
V = L.T.dot(X)
k = 200
X_tilde = L[:,0:k-1].dot(L[:,0:k-1].T).dot(X)
print np.allclose(X_tilde, X)

plt.imshow(X_tilde, cmap='gray')
plt.title('Approximated Image with full rank')
plt.show()

print (P/P.sum()).sum()
plt.plot((P/P.sum()).cumsum(), '-o')
plt.title('Cumulative Sum of Proportion of Total Variance')
plt.xlabel('index')
plt.ylabel('Proportion')
plt.show()

X_tilde_10 = L[:,0:10-1].dot(L[:,0:10-1].T).dot(X)
X_tilde_20 = L[:,0:20-1].dot(L[:,0:20-1].T).dot(X)
X_tilde_30 = L[:,0:30-1].dot(L[:,0:30-1].T).dot(X)
X_tilde_60 = L[:,0:60-1].dot(L[:,0:60-1].T).dot(X)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
ax1.imshow(X_tilde_10, cmap='gray')
ax1.set(title = 'Approximated Image with k = 10')
ax2.imshow(X_tilde_20, cmap='gray')
ax2.set(title = 'Approximated Image with k = 20')
ax3.imshow(X_tilde_30, cmap='gray')
ax3.set(title = 'Approximated Image with k = 30')
ax4.imshow(X_tilde_60, cmap='gray')
ax4.set(title = 'Approximated Image with k = 60')
ax5.imshow(X_tilde, cmap='gray')
ax5.set(title = 'Approximated Image with full rank')
ax6.imshow(X, cmap = 'gray')
ax6.set(title = 'Original Test Image of PCA')	
plt.show()