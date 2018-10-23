from sklearn.decomposition import PCA
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


def main():
    filter_show = range(5)  # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    print("Loading inputFeature_random ...")
    tar_data = torch.load('./conv_data/inputFeature_random.pkl')
    print("Loaded inputFeature_random !")
    print("Loading dW_random ...")
    dw_data = torch.load('./conv_data/dW_random.pkl')
    print("Loaded dW_random !")

    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([filter_num, num * top_num, count])
    dw_x = np.zeros([filter_num, num, count])
    all_x = np.zeros([filter_num, num * top_num + num, count])
    pca = {}
    for filter_order in filter_show:
        for order in range(num):
            for top_order in range(top_num):
                print("filter_order (%d/%d)\t" % (filter_order + 1, filter_num) + "order (%d/%d)\t" % (order + 1, num))
                tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][
                    top_order].numpy().reshape([1, -1])
                dw_x[filter_order][order] = dw_data[order][filter_order].numpy().reshape([1, -1])
        all_x[filter_order] = np.append(tar_x[filter_order], dw_x[filter_order], axis=0)

        print("\nfilter (%d/%d)\tall (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...")
        tmppca_tar = decomposition.PCA(whiten=True)
        tmppca_tar.fit(tar_x[filter_order])
        tmppca_dw = decomposition.PCA(whiten=True)
        tmppca_dw.fit(dw_x[filter_order])
        tmppca_all = decomposition.PCA(whiten=True)
        tmppca_all.fit(all_x[filter_order])
        pca[filter_order] = {'pca_tar': tmppca_tar, 'pca_dw': tmppca_dw}
        print("filter (%d/%d)\tall (%d)\t" % (filter_order + 1, filter_num, num) + "PCA Finished\n")

        # decent singular_values_ tar
        plt.figure(filter_order * 9)
        plt.plot(tmppca_tar.singular_values_, 'k', linewidth=2)
        plt.xlabel('n_components', fontsize=10)
        plt.ylabel('singular_values_', fontsize=10)
        plt.title("filter (%d/%d) tar (%d) " % (filter_order + 1, filter_num, num * top_num), fontsize=12)
        plt.savefig(
            "./res_random/decent/filter(%d,%d)tar(%d)decent" % (filter_order + 1, filter_num, num * top_num) + ".png")

        # decent singular_values_ dw
        plt.figure(filter_order * 9 + 1)
        plt.plot(tmppca_dw.singular_values_, 'k', linewidth=2)
        plt.xlabel('n_components', fontsize=10)
        plt.ylabel('singular_values_', fontsize=10)
        plt.title("filter (%d/%d) dw (%d) " % (filter_order + 1, filter_num, num), fontsize=12)
        plt.savefig("./res_random/decent/filter(%d,%d)dw(%d)decent" % (filter_order + 1, filter_num, num) + ".png")

        # decent singular_values_ all
        plt.figure(filter_order * 9 + 2)
        plt.plot(tmppca_all.singular_values_, 'k', linewidth=2)
        plt.xlabel('n_components', fontsize=10)
        plt.ylabel('singular_values_', fontsize=10)
        plt.title("filter (%d/%d) all (%d) " % (filter_order + 1, filter_num, num), fontsize=12)
        plt.savefig("./res_random/decent/filter(%d,%d)all(%d)decent" % (filter_order + 1, filter_num, num) + ".png")

        # histogram singular_values_ tar
        pl.figure(filter_order * 9 + 3)
        max_pca = max(tmppca_tar.singular_values_)
        min_pca = min(tmppca_tar.singular_values_)
        pl.hist(tmppca_tar.singular_values_, np.arange(min_pca, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) tar (%d) " % (filter_order + 1, filter_num, num * top_num), fontsize=12)
        plt.savefig("./res_random/hist/filter(%d,%d)tar(%d)histogram" % (
        filter_order + 1, filter_num, num * top_num) + ".png")

        # histogram singular_values_ dw
        pl.figure(filter_order * 9 + 4)
        max_pca = max(tmppca_dw.singular_values_)
        min_pca = min(tmppca_dw.singular_values_)
        pl.hist(tmppca_dw.singular_values_, np.arange(min_pca, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) dw (%d) " % (filter_order + 1, filter_num, num), fontsize=12)
        plt.savefig(
            "./res_random/hist/filter(%d,%d)dw(%d)histogram" % (filter_order + 1, filter_num, num * top_num) + ".png")

        # histogram singular_values_ all
        pl.figure(filter_order * 9 + 5)
        max_pca = max(tmppca_all.singular_values_)
        min_pca = min(tmppca_all.singular_values_)
        pl.hist(tmppca_all.singular_values_, np.arange(min_pca, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) all (%d) " % (filter_order + 1, filter_num, num), fontsize=12)
        plt.savefig("./res_random/hist/filter(%d,%d)all(%d)histogram" % (
        filter_order + 1, filter_num, num * top_num) + ".png")

        # histogram singular_values_ tar without 0
        pl.figure(filter_order * 9 + 6)
        max_pca = max(tmppca_tar.singular_values_)
        min_pca = min(tmppca_tar.singular_values_)
        pl.hist(tmppca_tar.singular_values_,
                np.arange((max_pca - min_pca) / interval, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) tar (%d) " % (filter_order + 1, filter_num, num * top_num), fontsize=12)
        plt.savefig("./res_random/hist_no_0/filter(%d,%d)tar(%d)histogram" % (
        filter_order + 1, filter_num, num * top_num) + ".png")

        # histogram singular_values_ dw without 0
        pl.figure(filter_order * 9 + 7)
        max_pca = max(tmppca_dw.singular_values_)
        min_pca = min(tmppca_dw.singular_values_)
        pl.hist(tmppca_dw.singular_values_,
                np.arange((max_pca - min_pca) / interval, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) dw (%d) " % (filter_order + 1, filter_num, num), fontsize=12)
        plt.savefig("./res_random/hist_no_0/filter(%d,%d)dw(%d)histogram" % (
        filter_order + 1, filter_num, num * top_num) + ".png")

        # histogram singular_values_ all without 0
        pl.figure(filter_order * 9 + 8)
        max_pca = max(tmppca_all.singular_values_)
        min_pca = min(tmppca_all.singular_values_)
        pl.hist(tmppca_all.singular_values_,
                np.arange((max_pca - min_pca) / interval, max_pca, (max_pca - min_pca) / interval))
        pl.ylabel('number_of_components', fontsize=10)
        pl.xlabel('singular_values_', fontsize=10)
        pl.title("filter (%d/%d) all (%d) " % (filter_order + 1, filter_num, num), fontsize=12)
        plt.savefig("./res_random/hist_no_0/filter(%d,%d)all(%d)histogram" % (
        filter_order + 1, filter_num, num * top_num) + ".png")

        # plt.show()
        print("Saving PCA...")
        torch.save(pca, 'pca_random.pth.tar')
        print("PCA saved.\n")


if __name__ == '__main__':
    main()