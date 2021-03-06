from sklearn.cluster import AgglomerativeClustering
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    filter_show = range(5)  # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    typeorder = 0
    typenum = 10
    data = {}
    print('Initiating...')
    tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_' + str(typeorder) + '_v1.pkl')
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    num *= typenum
    count = channel * filter_row * filter_col
    del tar_data
    np.set_printoptions(threshold=np.nan)
    print('Finished!')

    while typeorder < typenum:
        print('Loading Processed data(' + str(typeorder + 1) + '/' + str(typenum) + ')...')
        dir = './Processed_data/pretrain_' + str(typeorder) + '_v1/'
        datatmp = torch.load(dir + 'Processed_data_pretrain_' + str(typeorder) + '_v1_filter(0-9).pth.tar')
        if typeorder == 0:
            data = {}
            data['all'] = {}
        for key in datatmp:
            for filter_order in filter_show:
                if not key in data['all'].keys():
                    data['all'][key] = {}
                if not filter_order in data['all'][key].keys():
                    data['all'][key][filter_order] = datatmp[key][filter_order]
                else:
                    data['all'][key][filter_order] = np.concatenate(
                        [data['all'][key][filter_order], datatmp[key][filter_order]])
        print('Loaded data(' + str(typeorder + 1) + '/' + str(typenum) + ').')
        typeorder += 1

    for filter_order in filter_show:
        dir = "./Hierarchical_clustering_data/res_pretrain_%s/filter(%d,%d)/res/" % (
            'all_10', filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            continue
        f = open(dir + '/res.txt', 'w')
        print('Loading Processed data(' + 'all_10' + ')...', file=f)
        print('Loaded data(' + 'all_10' + ').', file=f)

        # PCA
        pca_val = {}
        pca_component = {}
        k = 0
        n = 3
        tmppca = decomposition.PCA()
        print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...", file=f)
        tmppca.fit(data['all']['tar'][filter_order] / count ** 0.5)  # 1
        pca_val["tar"] = tmppca.singular_values_
        pca_component["tar"] = tmppca.components_
        k += 1
        # print("(%d/%d) ..." % (k, n),file=f)
        tmppca.fit(data['all']['dw'][filter_order] / count ** 0.5)  # 2
        pca_val["dw"] = tmppca.singular_values_
        pca_component["dw"] = tmppca.components_
        k += 1
        # print("(%d/%d) ..." % (k, n),file=f)
        pca_val["dot_res"] = pca_val["dw"] * pca_val["tar"]  # 3
        pca_component["dot_res"] = pca_component["dw"] * pca_component["tar"]
        pca_component["dot_res_norm"] = pca_component["dot_res"] / np.mean(
            np.sum(pca_component["dot_res"] ** 2, axis=1)) ** 0.5
        k += 1
        # print("(%d/%d) ..." % (k, n),file=f)
        dir = "./Hierarchical_clustering_data/res_pretrain_%s/filter(%d,%d)/PCA/" % (
            'all_10', filter_order + 1, filter_num)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # torch.save(pca_val, dir + "pca_val_pretrain_%s_filter(%d,%d).pth.tar" % (
        #     'all_10', filter_order + 1, filter_num))
        # torch.save(pca_component, dir + "pca_component_%s_pretrain_filter(%d,%d).pth.tar" % (
        #     'all_10', filter_order + 1, filter_num))
        print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "PCA Finished", file=f)

        # Hierarchical_clustering
        select_feature_num = pca_component["dot_res_norm"].shape[1]
        select_feature = pca_component["dot_res_norm"]
        while select_feature_num > 0:
            print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num(%d)\t" % (
                filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...", file=f)
            print("\nfilter (%d/%d)\tsample (%d)\tselect_feature_num(%d)\t" % (
                filter_order + 1, filter_num, num, select_feature_num) + "Hierarchical_clustering ...")
            Cluster_data = {}
            # print("Corresponding dot result of the singular value:",file=f)
            # print(pca_val["dot_res"],file=f)
            # print("Corresponding tar singular value:",file=f)
            # print(pca_val["tar"],file=f)
            # print("Corresponding dw singular value:",file=f)
            # print(pca_val["dw"],file=f)

            for i in np.arange(1, 20):
                Cluster_data[i] = {}
                for method in {'complete', 'ward', 'average', 'single'}:
                    Cluster_data[i][method] = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                                                                      connectivity=None, linkage='complete',
                                                                      memory=None, n_clusters=i,
                                                                      pooling_func='deprecated').fit(select_feature)
                    print("Hierarchical_clustering(n_clusters=", i, ",linkage=" + method + ",select_feature=",
                          select_feature_num, "):",
                          file=f)
                    print("Hierarchical_clustering(n_clusters=", i, ",linkage=" + method + ",select_feature=",
                          select_feature_num, "):")
                    # print(Cluster_data[i][method].labels_)
                    unique, counts = np.unique(Cluster_data[i][method].labels_, return_counts=True)
                    res = dict(zip(unique, counts))
                    print(res, file=f)
                    print(res)
                    Cluster_data[i][method + '_res'] = res
                print("filter (%d/%d)\tsample (%d)\t" % (
                    filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished", file=f)
                print("filter (%d/%d)\tsample (%d)\t" % (
                    filter_order + 1, filter_num, num) + "Hierarchical_clustering Finished")

                print("Saving Kmeans...")
                dir = "./Hierarchical_clustering_data/res_pretrain_%s/filter(%d,%d)/select_feature(%d)/" % (
                'all_10', filter_order + 1, filter_num, select_feature_num)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                # torch.save(Cluster_data,
                           # dir + "Kmeans_%s_pretrain_filter(%d,%d).pth.tar" % ('all_10', filter_order + 1, filter_num))
                print("Kmeans saved.")

            select_feature_num -= 50
            if select_feature_num > 0:
                select_feature = select_feature[:, 0:select_feature_num]
                select_feature = select_feature / np.mean(np.sum(select_feature ** 2, axis=1)) ** 0.5


if __name__ == '__main__':
    main()