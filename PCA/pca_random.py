from sklearn.decomposition import PCA
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    filter_show = range(5)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    print("Loading inputFeature_random ...")
    tar_data = torch.load('./conv_data/inputFeature_random_MAX.pkl')
    print("Loaded inputFeature_random !")
    print("Loading dW_random ...")
    dw_data = torch.load('./conv_data/dW_random_MAX.pkl')
    print("Loaded dW_random !")

    print("Converting data ...")
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([max(filter_show)+1, num * top_num, count])
    dw_x = np.zeros([max(filter_show)+1, num, count])
    dot_tar_x = np.zeros([max(filter_show)+1, num * top_num, count])
    for filter_order in filter_show:
        for order in range(num):
            for top_order in range(top_num):
                if (order + 1) % np.ceil(num / 5) == 0:
                    print("filter_order (%d/%d)\t" %(filter_order + 1,filter_num)+"order (%d/%d)\t" %(order + 1,num))
                tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][top_order].numpy().reshape([1,-1])
                dw_x[filter_order][order] = dw_data[order][filter_order].numpy().reshape([1,-1])
        for top_order in range(top_num):
            dot_tar_x[filter_order][top_order * num:(top_order + 1) * num] = tar_x[filter_order][top_order * num:(top_order + 1) * num] * dw_x[filter_order][:]
    del tar_data
    del dw_data
    print("Convert finished !")

    for filter_order in filter_show:
        pca={}
        k = 0
        n = 9
        tmppca = decomposition.PCA()
        print("\nfilter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...")
        tmppca.fit(tar_x[filter_order] / count ** 0.5) # 1
        pca["tar"] = tmppca.singular_values_
        k += 1
        print("(%d/%d) ..." % (k, n))
        tmppca.fit(dw_x[filter_order] / count ** 0.5) # 2
        pca["dw"] = tmppca.singular_values_
        k += 1
        print("(%d/%d) ..." % (k, n))
        tmppca.fit(dot_tar_x[filter_order] / count ** 0.5) # 3
        pca["dot_tar"] = tmppca.singular_values_
        k += 1
        print("(%d/%d) ..." % (k, n))
        pca["dot_res"] = pca["dw"] * pca["tar"] # 4
        k += 1
        print("(%d/%d) ..." % (k, n))
        tar_x_norm = tar_x / np.mean(tar_x ** 2) ** 0.5 # 5
        k += 1
        print("(%d/%d) ..." % (k, n))
        dw_x_norm = dw_x / np.mean(dw_x ** 2) ** 0.5 # 6
        k += 1
        print("(%d/%d) ..." % (k, n))
        tmppca.fit(tar_x_norm[filter_order] / count ** 0.5) # 7
        pca["tar_norm"] = tmppca.singular_values_
        k += 1
        print("(%d/%d) ..." % (k, n))
        tmppca.fit(dw_x_norm[filter_order] / count ** 0.5) # 8
        pca["dw_norm"] = tmppca.singular_values_
        k += 1
        print("(%d/%d) ..." % (k, n))
        tmppca.fit(np.append(tar_x_norm[filter_order],dw_x_norm[filter_order],axis = 0) / count ** 0.5) # 9
        pca["all"] = tmppca.singular_values_
        k += 1
        print("(%d/%d) ..." % (k, n))
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished\n")

        # painting figures
        fig = 0
        for key in pca:
            # decent singular_values_
            plt.figure(fig)
            fig += 1
            plt.plot(pca[key], 'k', linewidth=2)
            plt.xlabel('n_components', fontsize=10)
            plt.ylabel('singular_values_', fontsize=10)
            plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) " % (num), fontsize=12)
            dir = "./res_random/decent/" + key + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir + "filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)decent" % (num) + ".png")

            # histogram singular_values_
            plt.figure(fig)
            fig += 1
            max_pca = max(pca[key])
            min_pca = min(pca[key])
            plt.hist(pca[key],np.arange(min_pca,max_pca,(max_pca - min_pca)/interval))
            plt.ylabel('number_of_components', fontsize=10)
            plt.xlabel('singular_values_', fontsize=10)
            plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) " % (num), fontsize=12)
            dir = "./res_random/hist/" + key + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir + "filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)hist" % (num) + ".png")

            # histogram singular_values_ tar without 0
            plt.figure(fig)
            fig += 1
            max_pca = max(pca[key])
            min_pca = min(pca[key])
            plt.hist(pca[key],np.arange((max_pca - min_pca)/interval,max_pca,(max_pca - min_pca)/interval))
            plt.ylabel('number_of_components', fontsize=10)
            plt.xlabel('singular_values_', fontsize=10)
            plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) no_0 " % (num), fontsize=12)
            dir = "./res_random/hist_no_0/" + key + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir + "filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)hist_no_0" % (num) + ".png")

        for i in range(fig):
            plt.figure(i).clear()
            # plt.show()

        print("Saving PCA...")
        dir = "./pca_data/filter(%d,%d)/" % (filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(pca,dir + "pca_random_filter(%d,%d).pth.tar" % (filter_order + 1, filter_num))
        print("PCA saved.\n")

if __name__ == '__main__':
    main()