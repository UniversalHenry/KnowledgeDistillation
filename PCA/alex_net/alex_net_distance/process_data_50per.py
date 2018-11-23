import torch
import numpy as np
import os

def main():

    filter_show = range(10)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    print('Loading inputFeature_pretrain ...')
    tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_0_CUB_sr0.5.pkl')
    print('Loaded inputFeature_pretrain !')
    print('Loading dW_pretrain ...')
    dw_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/dx_pretrain_0_CUB_sr0.5.pkl')
    print('Loaded dW_pretrain !')
    
    print('\tConverting data ...')
    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([filter_show.__len__(), num * top_num, count])
    dw_x = np.zeros([filter_show.__len__(), num, count])
    for filter_order in filter_show:
        for order in range(num):
            for top_order in range(top_num):
                if (order + 1) % np.ceil(num / 5) == 0:
                    print("\tfilter_order (%d/%d)\t" %(filter_order + 1,filter_num)+"order (%d/%d)\t" %(order + 1,num))
                tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][top_order].numpy().reshape([1,-1])
                dw_x[filter_order][order] = dw_data[order][filter_order].numpy().reshape([1,-1])
    print("\tConvert finished !")
    
    print('Saving Processed data...')
    data = {'tar':tar_x,'dw':dw_x}
    dir = './Processed_data/pretrain_0_CUB_sr0.5/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(data,dir + 'Processed_data_pretrain_0_CUB_sr0.5_filter(0-9).pth.tar')
    print('Processed data saved.\n')

if __name__ == '__main__':
    main()