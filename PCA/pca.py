from sklearn.decomposition import PCA
import torch
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab as pl

def main():
    np.random.seed(1)

    data = np.random.normal(loc=0.0, scale=10, size=(10, 512, 3, 3))
    shape = data.shape
    interval = 100
    num = shape[0]
    batch = shape[1]
    filter_col = shape[2]
    filter_row = shape[3]
    count = batch * filter_row * filter_col
    x=np.zeros([num ,count])
    for order in range(num):
        print("order (%d/%d)\t" %(order + 1,num))
        x[order] = data[order].reshape([1,-1])

    pca = decomposition.PCA(whiten=True)
    # pca = decomposition.PCA()
    pca.fit(x)
    print("all (%d)\t" %(num)+"PCA Finished")

    # decent explained_variance_
    plt.figure(0)
    plt.plot(pca.explained_variance_, 'k', linewidth=2)
    plt.xlabel('n_components', fontsize=10)
    plt.ylabel('explained_variance_', fontsize=10)
    plt.title("picture(%d)" %(num), fontsize=12)


    # decent singular_values_
    plt.figure(1)
    plt.plot(pca.singular_values_, 'k', linewidth=2)
    plt.xlabel('n_components', fontsize=10)
    plt.ylabel('singular_values_', fontsize=10)
    plt.title("picture(%d)" % (num), fontsize=12)

    # histogram explained_variance_
    pl.figure(2)
    max_pca = max(pca.explained_variance_)
    min_pca = min(pca.explained_variance_)
    pl.hist(pca.explained_variance_,np.arange(min_pca,max_pca,(max_pca - min_pca)/interval))
    pl.ylabel('number_of_components', fontsize=10)
    pl.xlabel('explained_variance_', fontsize=10)
    pl.title("picture(%d)" %(num), fontsize=12)

    # histogram singular_values_
    pl.figure(3)
    max_pca = max(pca.singular_values_)
    min_pca = min(pca.singular_values_)
    pl.hist(pca.singular_values_, np.arange(min_pca, max_pca, (max_pca - min_pca) / interval))
    pl.ylabel('number_of_components', fontsize=10)
    pl.xlabel('singular_values_', fontsize=10)
    pl.title("picture(%d)" % (num), fontsize=12)

    plt.show()

if __name__ == '__main__':
    main()