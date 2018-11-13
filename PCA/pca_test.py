# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:14:36 2018

@author: lzn
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition

# def esd(X):
#     n = X.shape[1] # sample numbers
#     _, D, _ = np.linalg.svd(X / n**0.5)
#     plt.figure()
#     plt.hist(D[:], normed=True) # density distribution
#     plt.show()
#     plt.close()
np.random.seed(1)
X = np.random.randn(500,1000)
m, n = X.shape # sample numbers
_, D, _ = np.linalg.svd(X / n**0.5)

tmppca = decomposition.PCA(svd_solver = 'auto')
# pca = decomposition.PCA()
tmppca.fit(X / n ** 0.5)

d, _  = np.linalg.eigh(np.dot(X, X.T) / n)
if m > n:
    svs = D[m-n:n,]

max_pca = max(D)
min_pca = min(D)
plt.figure(0)
plt.hist(D[:],np.arange((max_pca - min_pca) / 100, max_pca, (max_pca - min_pca) / 100))

plt.figure(1)
# max_pca = max(tmppca.singular_values_)
# min_pca = min(tmppca.singular_values_)
plt.hist(tmppca.singular_values_,np.arange((max_pca - min_pca) / 100, max_pca, (max_pca - min_pca) / 100))
plt.show()
plt.close()