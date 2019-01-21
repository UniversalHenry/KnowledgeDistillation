import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def genPhoto(d1,d2):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    df1 = pd.DataFrame(d1, columns=["0","1","2"])
    df2 = pd.DataFrame(d2, columns=["0","1","2"])
    df1.plot.area(colormap="OrRd", alpha=1, ax=axes[0])
    df2.plot.area(colormap="OrRd", alpha=1, ax=axes[1])
    a1 = np.average(d1,axis = 0)
    a2 = np.average(d2,axis = 0)
    axes[0].set_title('v1_(0,1,2)='+ a1.__str__())
    axes[1].set_title('v2_(0,1,2)='+ a2.__str__())
    plt.savefig('./data/result.jpg')

def main():
    data1 = torch.load('/data/HaoChen/knowledge_distillation/FeatureFactorization/contrib_collect_v4.pkl')
    data2 = torch.load('/data/HaoChen/knowledge_distillation/FeatureFactorization/contrib_collect_Lu.pkl')
    data1 = data1.data.numpy()
    data2 = data2.data.numpy()
    plt.switch_backend('agg')
    genPhoto(data1,data2)

main()