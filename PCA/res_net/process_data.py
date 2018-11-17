import torch
import numpy as np
import os

def main():

    filter_show = range(10)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    typeorder = 0
    typenum = 10
    while typeorder < typenum:
        print('Loading inputFeature_pretrain('+ str(typeorder + 1) + '/'+ str(typenum) + ') ...')
        tar_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/tar_pretrain_'+ str(typeorder) + '_v1.pkl')
        print('Loaded inputFeature_pretrain('+ str(typeorder + 1) + '/'+ str(typenum) + ') !')
        print('Loading dW_pretrain ('+ str(typeorder + 1) + '/'+ str(typenum) + ')...')
        dw_data = torch.load('/data/HaoChen/knowledge_distillation/PCA/dx_pretrain_'+ str(typeorder) + '_v1.pkl')
        print('Loaded dW_pretrain('+ str(typeorder + 1) + '/'+ str(typenum) + ') !')
    
        print('\tConverting data('+ str(typeorder + 1) + '/'+ str(typenum) + ') ...')
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
        
        print('Saving Processed data('+ str(typeorder + 1) + '/'+ str(typenum) + ')...')
        data = {'tar':tar_x,'dw':dw_x}
        dir = './Processed_data/pretrain_'+ str(typeorder) + '_v1/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(data,dir + 'Processed_data_pretrain_'+ str(typeorder) + '_v1_filter(0-9).pth.tar')
        print('Processed data('+ str(typeorder + 1) + '/'+ str(typenum) + ') saved.\n')
        
        typeorder += 1
        
if __name__ == '__main__':
    main()