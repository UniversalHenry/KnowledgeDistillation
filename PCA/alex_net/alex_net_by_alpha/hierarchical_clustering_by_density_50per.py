import torch
import scipy.cluster.hierarchy as sch
from sklearn import decomposition
import numpy as np
import os
import re

def main():
    target_dir = '/data/HaoChen/knowledge_distillation/PCA/select_fig/tar_pretrain_0_CUB_sr05'
    select_feature_max = {}
    for _, _, files in os.walk(target_dir):
        for the_filename in files:
            pattern = re.compile(r'\d+')
            find_val = pattern.findall(the_filename)
            assert (find_val.__len__() == 2)
            select_feature_max.update({int(find_val[0]):int(find_val[1])})
    filter_show = range(5)   # which filters to calculate
    # where to load the data
    data = {}
    typeorder = 0
    print('Initiating...')
    tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_0_CUB_sr0.5.pkl')
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    del tar_data
    np.set_printoptions(threshold=np.nan)
    print('Finished!')

    print('Loading Processed data...')
    dir = './Processed_data/pretrain_0_CUB_sr0.5/'
    data[typeorder] = torch.load( dir + 'Processed_data_pretrain_0_CUB_sr0.5_filter(0-9).pth.tar')
    print('Loaded data.')

    for filter_order in filter_show:
        dir = "./Hierarchical_data/res_0.5_parttrain_%d/filter(%d,%d)/res/" % (
            typeorder, filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # else:
        #     continue
        f = open(dir + '/res.txt', 'w')
        print('Loading Processed data...')
        print('Loaded data.')

        # PCA
        pca_val={}
        pca_component={}
        tmppca = decomposition.PCA()
        print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...")
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
        print("filter (%d/%d)\tsample (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished")

        # Hierarchical_clustering

        Cluster_data= {}
        alpha_range = np.arange(0.01,1.01,0.01)
        rate_range = np.arange(0,1,0.01)
        for method in ['centroid','complete', 'average', 'single', 'ward']:
            Cluster_data.update({method: np.zeros((rate_range.__len__(),alpha_range.__len__()))})
            for j in np.arange(alpha_range.__len__()):
                alpha = alpha_range[j]
                select_feature_num = int(np.ceil(select_feature_max[filter_order] * alpha))
                select_feature = pca_component["dot_res_norm"][:, 0:select_feature_num]
                select_feature = select_feature / np.mean(np.sum(select_feature ** 2, axis=1)) ** 0.5
                disMat = sch.distance.pdist(select_feature, 'euclidean')
                print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num (%d)\t" % (
                    filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...", file=f)
                print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num (%d)\t" % (
                    filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...")
                Z = sch.linkage(disMat, method=method)
                maxd = Z[-1][2]
                # find the min distance
                for i in np.arange(rate_range.__len__()):
                    rate = rate_range[i]
                    tol = maxd * rate
                    cluster = sch.fcluster(Z, t=tol, criterion='distance')
                    unique, counts = np.unique(cluster, return_counts=True)
                    Cluster_data[method][i][j] = max(unique)
                    # res = dict(zip(unique, counts))
                    print("Hierarchical_clustering(rate=%.2f" % rate,",alpha=%.2f"% alpha,",tol=%.3f" % tol,",n_cluster=", max(unique),
                          ",max_counts=",max(counts),",linkage="+ method+ ",select_feature=",select_feature_num,")", file=f)
                    print("Hierarchical_clustering(rate=%.2f" % rate,",alpha=%.2f"% alpha, ",tol=%.3f" % tol, ",n_cluster=", max(unique),
                          ",max_counts=",max(counts),",linkage=" + method +  ",select_feature=", select_feature_num, ")")
        print("filter (%d/%d)\tsample (%d)\t" % (
            filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished", file=f)
        print("filter (%d/%d)\tsample (%d)\t" % (
            filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished")
        print('')
        torch.save(Cluster_data,dir + "cluster_data.pth.tar")

if __name__ == '__main__':
    main()