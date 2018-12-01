import torch
import scipy.cluster.hierarchy as sch
import numpy as np


def main():
    # a simple sample of hierarchical_clustering (criterion is distance)

    sample_num, dimension = 1000, 20
    input_data = np.random.random(
        (sample_num, sample_num))  # assume the input is 1000 random samples with 20 dimensions of features
    Cluster_data = {}
    for method in ['centroid', 'complete', 'average', 'single', 'ward']:  # for all method you may easily apply
        disMat = sch.distance.pdist(input_data, 'euclidean')
        Z = sch.linkage(disMat, method=method)
        maxd = Z[-1][2]  # for finding the min distance to cluster into one catagory
        rate = 0.5  # assume looking for rate with 0.5
        tol = rate * maxd
        cluster = sch.fcluster(Z, t=tol, criterion='distance')  # assume criterion is distance
        Cluster_data.update({method: cluster})
    torch.save(Cluster_data, "./cluster_data.pth.tar")


if __name__ == '__main__':
    main()