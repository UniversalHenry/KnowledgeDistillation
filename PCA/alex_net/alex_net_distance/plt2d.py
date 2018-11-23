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
    for root_dir in ["./Hierarchical_density_data/res_0.2_parttrain",
                     "./Hierarchical_density_data/res_0.5_parttrain",
                "./Hierarchical_density_data/res_pretrain"]:
        for filter_order in filter_show:
            print(root_dir)
            print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "Ploting ...")
            dir = root_dir + "_%d/filter(%d,%d)/res/" % (
                typeorder, filter_order + 1, filter_num)
            assert (os.path.exists(dir))
            Cluster_data = torch.load(dir + "cluster_data.pth.tar")
            dir = dir + "fig_2d/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            # else:
            #     continue
            x = np.linspace(count % 50, count , int(count / 50) + 1)
            for j in np.linspace(10, 90, 9):
                rate = j / 100
                for key in Cluster_data.keys():
                    print(key, '\tShape:', Cluster_data[key].shape , '\tRate:',rate)
                    z = Cluster_data[key]
                    y = np.zeros(int(count / 50) + 1)
                    for i in np.arange(z.shape[1]):
                        y[i] = z[int(j)][i]
                    fig = plt.figure()
                    plt.plot(x,y)
                    plt.title(key + "_filter(%d/%d)sample(%d)_rate(%f)" % (filter_order + 1, filter_num, num, rate), color='b')
                    tmpdir = dir + ('rate_%.1f/' % rate)
                    if not os.path.exists(tmpdir):
                        os.makedirs(tmpdir)
                    fig.savefig(tmpdir + key + '.png')
            print("filter (%d/%d)\tsample (%d)\t" % (filter_order + 1, filter_num, num) + "Ploting Finished")
            print("")

if __name__ == '__main__':
    main()