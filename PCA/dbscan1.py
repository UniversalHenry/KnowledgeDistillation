from sklearn.cluster import DBSCAN
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    filter_show = range(5)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    print("Loading inputFeature_pretrain ...")
    tar_data = torch.load('./conv_data/inputFeature_pretrain_MAX.pkl')
    print("Loaded inputFeature_pretrain !")
    print("Loading dW_pretrain ...")
    dw_data = torch.load('./conv_data/dW_pretrain_MAX.pkl')
    print("Loaded dW_pretrain !")

    print("Converting data ...")
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([filter_show.__len__(), num * top_num, count])
    dw_x = np.zeros([filter_show.__len__(), num, count])
    dot_tar_x = np.zeros([filter_show.__len__(), num * top_num, count])
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
        pca_val={}
        pca_component={}
        k = 0
        n = 3
        tmppca = decomposition.PCA()
        print("\nfilter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...")
        tmppca.fit(tar_x[filter_order] / count ** 0.5) # 1
        pca_val["tar"] = tmppca.singular_values_
        pca_component["tar"] = tmppca.components_
        k += 1
        print("(%d/%d) ..." % (k, n))
        tmppca.fit(dw_x[filter_order] / count ** 0.5) # 2
        pca_val["dw"] = tmppca.singular_values_
        pca_component["dw"] = tmppca.components_
        k += 1
        print("(%d/%d) ..." % (k, n))
        pca_val["dot_res"] = pca_val["dw"] * pca_val["tar"] # 3
        pca_component["dot_res"] = pca_component["dw"] * pca_component["tar"]
        pca_component["dot_res_norm"] = pca_component["dot_res"] / np.mean(np.sum(pca_component["dot_res"] ** 2,axis=1)) ** 0.5
        k += 1
        print("(%d/%d) ..." % (k, n))
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished\n")

        # dbscan
        print("\nfilter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "DBSCAN ...")
        dbscan = {}
        n = 0
        print("Corresponding dot result of the singular value:")
        print(pca_val["dot_res"])
        print("Corresponding tar singular value:")
        print(pca_val["tar"])
        print("Corresponding dw singular value:")
        print(pca_val["dw"])
        for i in np.arange(0.1,3,0.1):
            dbscan[n] = DBSCAN(eps=i,min_samples=2).fit(pca_component["dot_res_norm"])
            print("DBSCAN(", n, ",eps=",i,",min_samples=2):")
            print(dbscan[n].labels_)
            n+=1
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"DBSCAN Finished\n")

        print("Saving dbscan...")
        dir = "./dbscan_data/res_pretrain/filter(%d,%d)/" % (filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(dbscan,dir + "dbscan_pretrain_filter(%d,%d).pth.tar" % (filter_order + 1, filter_num))
        torch.save(pca_val, dir + "pca_val_pretrain_filter(%d,%d).pth.tar" % (filter_order + 1, filter_num))
        torch.save(pca_component, dir + "pca_component_pretrain_filter(%d,%d).pth.tar" % (filter_order + 1, filter_num))
        print("Dbscan saved.\n")

if __name__ == '__main__':
    main()