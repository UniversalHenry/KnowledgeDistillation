
# coding: utf-8

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import torch
import os


def extract_conv_weight(model_path = 'model_best_CUB__Z.pth.tar', layer = 10):  # for cifar layer = 28
    net = torch.load(model_path)
    w = net['state_dict']['features.{}.weight'.format(layer)].cpu()
    del net
    return w


model_path = 'model_best_CUB__poorData.pth.tar'

w = extract_conv_weight(model_path)
num_filter = w.size()[0]
filters = w.numpy().reshape(num_filter,-1)

tmp = model_path.split('.')
model = ''
for t in tmp[:-2]:
    model += t

save_path = './W_distribution/'+model+'/'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(20):
    n, bins, patches = plt.hist((filters[134,:],), bins=50,rwidth = 0.8, histtype='barstacked',alpha = 0.5, stacked=0)
    plt.title(r'$Num of filter: {}/{}$ model: {}'.format(i,num_filter,model))
    plt.savefig(save_path+'{}.png'.format(i))



