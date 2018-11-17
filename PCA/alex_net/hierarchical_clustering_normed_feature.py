from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    filter_show = range(5)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    data = {}
    typeorder = 0
    print('Initiating...')
    tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_0_CUB_v1.pkl')
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    del tar_data
    np.set_printoptions(threshold=np.nan)
    print('Finished!')

    print('Loading Processed data...')
    dir = './Processed_data/pretrain_0_CUB_v1/'
    data[typeorder] = torch.load( dir + 'Processed_data_pretrain_0_CUB_v1_filter(0-9).pth.tar')
    print('Loaded data.')

    for filter_order in filter_show:
        dir = "./Hierarchical_clustering_data_normed_feature/res_pretrain_%d/filter(%d,%d)/res/" % (
            typeorder, filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            continue
        f = open(dir + '/res.txt', 'w')
        print('Loading Processed data...',file=f)
        print('Loaded data.',file=f)

        # PCA
        pca_val={}
        pca_component={}
        k = 0
        n = 3
        tmppca = decomposition.PCA()
        print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...",file=f)
        tmppca.fit(data[typeorder]['tar'][filter_order] / count ** 0.5) # 1
        pca_val["tar"] = tmppca.singular_values_
        pca_component["tar"] = tmppca.components_
        k += 1
        # print("(%d/%d) ..." % (k, n),file=f)
        tmppca.fit(data[typeorder]['dw'][filter_order] / count ** 0.5) # 2
        pca_val["dw"] = tmppca.singular_values_
        pca_component["dw"] = tmppca.components_
        k += 1
        # print("(%d/%d) ..." % (k, n),file=f)
        pca_val["dot_res"] = pca_val["dw"] * pca_val["tar"] # 3
        pca_component["dot_res"] = pca_component["dw"] * pca_component["tar"]
        pca_component["dot_res_norm"] = preprocessing.scale(pca_component["dot_res"])
        pca_component["dot_res_norm"] = pca_component["dot_res_norm"] / np.mean(np.sum(pca_component["dot_res_norm"] ** 2,axis=1)) ** 0.5
        k += 1
        # print("(%d/%d) ..." % (k, n),file=f)
        dir = "./Hierarchical_clustering_data_normed_feature/res_pretrain_%d/filter(%d,%d)/PCA/" % (
            typeorder, filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # torch.save(pca_val, dir + "pca_val_pretrain_%d_filter(%d,%d).pth.tar" % (typeorder, filter_order + 1, filter_num))
        # torch.save(pca_component, dir + "pca_component_%d_pretrain_filter(%d,%d).pth.tar" % (typeorder, filter_order + 1, filter_num))
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished",file=f)

        # Hierarchical_clustering
        select_feature_num = pca_component["dot_res_norm"].shape[1]
        select_feature = pca_component["dot_res_norm"]
        while select_feature_num > 0:
            print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num(%d)\t" % (
                filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...",file=f)
            print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num(%d)\t" % (
                filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...")
            Cluster_data = {}
            # print("Corresponding dot result of the singular value:",file=f)
            # print(pca_val["dot_res"],file=f)
            # print("Corresponding tar singular value:",file=f)
            # print(pca_val["tar"],file=f)
            # print("Corresponding dw singular value:",file=f)
            # print(pca_val["dw"],file=f)

            for i in np.arange(1,20):
                Cluster_data[i] = {}
                for method in {'complete','ward','average','single'}:
                    Cluster_data[i][method] = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, linkage='complete', memory=None, n_clusters=i,
                        pooling_func='deprecated').fit(select_feature)
                    print("Hierarchical_clustering(n_clusters=",i,",linkage="+ method+ ",select_feature=",select_feature_num,"):",
                          file=f)
                    print("Hierarchical_clustering(n_clusters=", i, ",linkage=" + method + ",select_feature=",
                          select_feature_num, "):")
                    # print(Cluster_data[i][method].labels_)
                    unique, counts = np.unique(Cluster_data[i][method].labels_, return_counts=True)
                    res = dict(zip(unique, counts))
                    print(res,file=f)
                    print(res)
                    Cluster_data[i][method + '_res'] = res
                print("filter (%d/%d)\tsample (%d)\t" %(
                    filter_order + 1, filter_num, num)+"Hierarchical_clustering Finished",file=f)
                print("filter (%d/%d)\tsample (%d)\t" % (
                    filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished")

                print("Saving Hierarchical_clustering...")
                dir = "./Hierarchical_clustering_data_normed_feature/res_pretrain_%d/filter(%d,%d)/select_feature(%d)/" % ( typeorder ,filter_order + 1, filter_num ,select_feature_num)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                # # torch.save(Cluster_data,dir + "Hierarchical_clustering_%d_pretrain_(%s)_filter(%d,%d).pth.tar" % (typeorder ,method ,filter_order + 1, filter_num))
                print("Hierarchical_clustering saved.")

            select_feature_num -= 50
            if select_feature_num>0:
                select_feature = select_feature[:, 0:select_feature_num]
                select_feature = select_feature / np.mean(np.sum(select_feature ** 2, axis=1)) ** 0.5


if __name__ == '__main__':
    main()