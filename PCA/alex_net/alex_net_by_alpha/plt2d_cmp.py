import torch
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import os

def main():
    print('Initiating...')
    filter_show = range(5)
    typeorder = 0
    tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_0_CUB_v1.pkl')
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    plt.switch_backend('agg')
    del tar_data
    print('Finished!')

    cmp_dir = "./Hierarchical_data/compare"
    alpha_range = np.arange(0.01, 1.01, 0.01)
    rate_range = np.arange(0, 1, 0.01)
    x = rate_range
    for filter_order in filter_show:
        for key in ['centroid', 'complete', 'average', 'single', 'ward']:
            for j in np.arange(0,alpha_range.__len__()):
                alpha = alpha_range[j]
                y = {}
                for root_dir in ["./Hierarchical_data/res_0.2_parttrain",
                                 "./Hierarchical_data/res_0.5_parttrain",
                                 "./Hierarchical_data/res_pretrain"]:
                    print(root_dir)
                    print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "Ploting ...")
                    dir = root_dir + "_%d/filter(%d,%d)/res/" % (
                        typeorder, filter_order + 1, filter_num)
                    assert (os.path.exists(dir))
                    Cluster_data = torch.load(dir + "cluster_data.pth.tar")
                    print(key, '\tShape:', Cluster_data[key].shape , '\tAlpha:%.2f'% alpha)
                    z = Cluster_data[key]
                    y.update({root_dir:np.zeros(x.__len__())})
                    for i in np.arange(y[root_dir].__len__()):
                        y[root_dir][i] = z[i][j]
                fig = plt.figure()
                plt.plot(x,y["./Hierarchical_data/res_0.2_parttrain"],'r',x,y["./Hierarchical_data/res_0.5_parttrain"], 'g',
                         x,y["./Hierarchical_data/res_pretrain"], 'b')
                plt.title(key + "_filter(%d/%d)sample(%d)_alpha(%.2f)\n" %
                          (filter_order + 1, filter_num, num, alpha)+ "0.2_red,0.5_green,1_blue"
                          , color='b')
                plt.xlabel('rate',color='b')
                plt.ylabel('n_cluster',color='b')
                savedir = cmp_dir + "_%d/filter(%d,%d)/res/" % (
                        typeorder, filter_order + 1, filter_num)
                tmpdir = savedir + ('alpha_%.2f/' % alpha)
                if not os.path.exists(tmpdir):
                    os.makedirs(tmpdir)
                fig.savefig(tmpdir + key + '.png')
        print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "Ploting Finished")
    print("")

if __name__ == '__main__':
    main()