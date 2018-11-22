import torch
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
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
        dir = "./Hierarchical_density_data/res_pretrain_%d/filter(%d,%d)/res/" % (
            typeorder, filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # else:
        #     continue
        f = open(dir + '/res.txt', 'w')
        print('Loading Processed data...',file=f)
        print('Loaded data.',file=f)

        # PCA
        pca_val={}
        pca_component={}
        tmppca = decomposition.PCA()
        print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...",file=f)
        tmppca.fit(data[typeorder]['tar'][filter_order] / count ** 0.5)
        pca_val["tar"] = tmppca.singular_values_
        pca_component["tar"] = tmppca.components_
        tmppca.fit(data[typeorder]['dw'][filter_order] / count ** 0.5)
        pca_val["dw"] = tmppca.singular_values_
        pca_component["dw"] = tmppca.components_
        pca_val["dot_res"] = pca_val["dw"] * pca_val["tar"]
        pca_component["dot_res"] = pca_component["dw"] * pca_component["tar"]
        pca_component["dot_res_norm"] = pca_component["dot_res"] / np.mean(
            np.sum(pca_component["dot_res"] ** 2, axis=1)) ** 0.5
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished",file=f)

        # Hierarchical_clustering

        Cluster_data= {}
        for method in {'complete', 'average', 'single', 'ward'}:
            select_feature_num = pca_component["dot_res_norm"].shape[1]
            select_feature = pca_component["dot_res_norm"]
            Cluster_data.update({method: np.zeros((100,int(select_feature_num / 50) + 1))})
            while select_feature_num > 0:
                disMat = sch.distance.pdist(select_feature, 'euclidean')
                print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num(%d)\t" % (
                    filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...",file=f)
                print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num(%d)\t" % (
                    filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...")
                Z = sch.linkage(disMat, method=method)
                maxd = max(disMat)
                mind = 0
                while abs(maxd - mind) > 1e-4:
                    midd = (maxd + mind) / 2
                    n_cluster = max(sch.fcluster(Z, t=midd))
                    if n_cluster > 1:
                        mind = midd
                    else:
                        maxd = midd
                # find the min distance
                for i in np.arange(0, 100):
                    rate = 1 - i / 100
                    tol = maxd * rate
                    cluster = sch.fcluster(Z, t=tol)
                    unique, counts = np.unique(cluster, return_counts=True)
                    Cluster_data[method][i][int(select_feature_num / 50)] = max(unique)
                    res = dict(zip(unique, counts))
                    print("Hierarchical_clustering(rate=%.2f" % rate,",tol=%.3f" % tol,",n_cluster=", max(unique),
                          ",max_counts=",max(counts),",linkage="+ method+ ",select_feature=",select_feature_num,")",
                          file=f)
                    print("Hierarchical_clustering(rate=%.2f" % rate, ",tol=%.3f" % tol, ",n_cluster=", max(unique),
                          ",max_counts=",max(counts),",linkage=" + method + ",select_feature=", select_feature_num, ")")

                select_feature_num -= 50
                if select_feature_num>0:
                    select_feature = select_feature[:, 0:select_feature_num]
                    select_feature = select_feature / np.mean(np.sum(select_feature ** 2, axis=1)) ** 0.5
        print("filter (%d/%d)\tsample (%d)\t" % (
            filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished", file=f)
        print("filter (%d/%d)\tsample (%d)\t" % (
            filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished")
        torch.save(Cluster_data,dir + "cluster_data.pth.tar")

if __name__ == '__main__':
    main()