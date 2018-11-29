from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    filter_show = range(20)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    print("Loading inputFeature_pretrain ...")
    tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_0_poorData.pkl')
    print("Loaded inputFeature_pretrain !")
    print("Loading dW_pretrain ...")
    dw_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/dx_pretrain_0_poorData.pkl')
    print("Loaded dW_pretrain !")

    print("Converting data ...")
    np.set_printoptions(threshold=1e6);
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([filter_show.__len__(), num * top_num, count])
    dw_x = np.zeros([filter_show.__len__(), num, count])
    for filter_order in filter_show:
        for order in range(num):
            for top_order in range(top_num):
                if (order + 1) % np.ceil(num / 5) == 0:
                    print("filter_order (%d/%d)\t" %(filter_order + 1,filter_num)+"order (%d/%d)\t" %(order + 1,num))
                tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][top_order].numpy().reshape([1,-1])
                dw_x[filter_order][order] = dw_data[order][filter_order].numpy().reshape([1,-1])
    del tar_data
    del dw_data
    print("Convert finished !")

    plt.switch_backend('agg')

    for filter_order in filter_show:
        pca={}
        k = 0
        n = 3
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
        pca["dot_res"] = pca["dw"] * pca["tar"] # 3
        k += 1
        print("(%d/%d) ..." % (k, n))
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished\n")

        # painting figures
        fig = 0
        for key in pca:
            print('DBSCAN ...')
            dbscan_data = DBSCAN(eps=0.001 * (pca[key].max() - pca[key].min()), min_samples=2).fit(
                pca[key].reshape((-1, 1)))
            _, counts = np.unique(dbscan_data.labels_, return_counts=True)
            effective_dimension = sum(counts) - max(counts)
            print(key,'effective dimension:',effective_dimension)
            print('DBSCAN finished!')

        #     # yx_line singular_values_ tar
        #     plt.figure(fig)
        #     fig += 1
        #     plt.axis([min(pca[key]), max(pca[key]), min(pca[key]),max(pca[key])])
        #     plt.scatter(pca[key],pca[key],marker='x')
        #     plt.ylabel('singular_values_', fontsize=10)
        #     plt.xlabel('singular_values_', fontsize=10)
        #     plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) no_0 " % (num), fontsize=12)
        #     dir = "./res_pretrain_poor/yx_line/" + key + "/"
        #     if not os.path.exists(dir):
        #         os.makedirs(dir)
        #     plt.savefig(dir + "filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)yx_line" % (num) + ".png")
        #
        # for i in range(fig):
        #     plt.figure(i).clear()
        #     # plt.show()

        # print("Saving PCA...")
        # dir = "./pca_data/res_pretrain_poor/filter(%d,%d)/" % (filter_order + 1, filter_num)
        # if not os.path.exists(dir):
        #     os.makedirs(dir)
        # torch.save(pca,dir + "pca_pretrain_filter(%d,%d).pth.tar" % (filter_order + 1, filter_num))
        # print("PCA saved.\n")

if __name__ == '__main__':
    main()