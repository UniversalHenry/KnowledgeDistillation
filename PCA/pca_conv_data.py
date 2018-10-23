from sklearn.decomposition import PCA
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def main():

    filter_show = [0]   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    dict = torch.load('./conv_data/fdata.pkl')  # where to load the data

    tar_data = dict['tar_data']
    dw_data = dict['dw_data']
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([filter_num, num * top_num, count])
    dw_x = np.zeros([filter_num, num, count])
    pca = {}
    for filter_order in filter_show:
        for order in range(num):
            for top_order in range(top_num):
                print("filter_order (%d/%d)\t" %(filter_order + 1,filter_num)+"order (%d/%d)\t" %(order + 1,num))
                tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][top_order].numpy().reshape([1,-1])
                dw_x[filter_order][order] = dw_data[order][filter_order].numpy().reshape([1,-1])
        print("\nfilter (%d/%d)\tall (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...")
        tmppca_tar = decomposition.PCA(whiten=True)
        tmppca_tar.fit(tar_x[filter_order])
        tmppca_dw = decomposition.PCA(whiten=True)
        tmppca_dw.fit(dw_x[filter_order])
        pca[filter_order] = {'pca_tar':tmppca_tar,'pca_dw':tmppca_dw}
        print("filter (%d/%d)\tall (%d)\n" %(filter_order + 1, filter_num, num)+"PCA Finished")

        # decent singular_values_ tar
        plt.figure(filter_order * 4)
        plt.plot(tmppca_tar.singular_values_, 'k', linewidth=2)
        plt.xlabel('n_components', fontsize=10)
        plt.ylabel('singular_values_', fontsize=10)
        plt.title("filter (%d/%d) tar all (%d) " % (filter_order + 1, filter_num, num*top_num), fontsize=12)

        # decent singular_values_ dw
        plt.figure(filter_order * 4 + 1)
        plt.plot(tmppca_dw.singular_values_, 'k', linewidth=2)
        plt.xlabel('n_components', fontsize=10)
        plt.ylabel('singular_values_', fontsize=10)
        plt.title("filter (%d/%d) dw all (%d) " % (filter_order + 1, filter_num, num), fontsize=12)

        # histogram singular_values_ tar
        pl.figure(filter_order * 4 + 2)
        max_pca = max(tmppca_tar.singular_values_)
        min_pca = min(tmppca_tar.singular_values_)
        pl.hist(tmppca_tar.singular_values_,np.arange(min_pca,max_pca,(max_pca - min_pca)/interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) tar all (%d) " % (filter_order + 1, filter_num, num*top_num), fontsize=12)

        # histogram singular_values_ dw
        pl.figure(filter_order * 4 + 3)
        max_pca = max(tmppca_dw.singular_values_)
        min_pca = min(tmppca_dw.singular_values_)
        pl.hist(tmppca_dw.singular_values_, np.arange(min_pca, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) dw all (%d) " % (filter_order + 1, filter_num, num), fontsize=12)

        plt.show()
    print("Saving PCA...")
    torch.save(pca,'pca.pth.tar')
    print("PCA saved.")

if __name__ == '__main__':
    main()