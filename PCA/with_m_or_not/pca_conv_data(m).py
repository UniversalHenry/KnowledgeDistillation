from sklearn.decomposition import PCA
import torch
from sklearn.cluster import DBSCAN
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    filter_show = range(30)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    for name in ['pretrain_m_RFx_full','pretrain_m_RFx_init','pretrain_m_RFx_num4','RFx_full','RFx_init','RFx_num4']:
        print('Loading inputFeature_pretrain ...')
        tar_data = torch.load('/data/HaoChen/knowledge_distillation/Overfit/TryAgain/tar_'+ name + '.pkl')
        print(name)
        print('Loaded inputFeature_pretrain !')
        print('Loading dx_pretrain ...')
        dx_data = torch.load('/data/HaoChen/knowledge_distillation/Overfit/TryAgain/dx_'+ name + '.pkl')
        print('Loaded dx_pretrain !')
    
        print('Converting data ...')
        num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
        count = channel * filter_row * filter_col
        tar_x = np.zeros([filter_show.__len__(), num * top_num, count])
        dx_x = np.zeros([filter_show.__len__(), num, count])
        for filter_order in filter_show:
            for order in range(num):
                for top_order in range(top_num):
                    if (order + 1) % np.ceil(num / 5) == 0:
                        print('filter_order (%d/%d)\t' %(filter_order + 1,filter_num)+'order (%d/%d)\t' %(order + 1,num))
                    tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][top_order].numpy().reshape([1,-1])
                    dx_x[filter_order][order] = dx_data[order][filter_order].numpy().reshape([1,-1])
        del tar_data
        del dx_data
        print('Convert finished !')
    
        plt.switch_backend('agg')
    
        for filter_order in filter_show:
            pca={}
            k = 0
            n = 6
            tmppca = decomposition.PCA()
            print(name)
            print('filter (%d/%d)\tsample (%d)\t' % (filter_order + 1, filter_num, num) + 'PCA ...')
            tmppca.fit(tar_x[filter_order] / count ** 0.5) # 1
            pca['tar'] = tmppca.singular_values_
            k += 1
            print('(%d/%d) ...' % (k, n))
            tmppca.fit(dx_x[filter_order] / count ** 0.5) # 2
            pca['dx'] = tmppca.singular_values_
            k += 1
            print('(%d/%d) ...' % (k, n))
            pca['dot_res'] = pca['dx'] * pca['tar'] # 3
            k += 1
            print('(%d/%d) ...' % (k, n))
            tar_x_norm = tar_x / np.mean(tar_x ** 2) ** 0.5
            dx_x_norm = dx_x / np.mean(dx_x ** 2) ** 0.5
            tmppca.fit(tar_x_norm[filter_order] / count ** 0.5) # 4
            pca['tar_norm'] = tmppca.singular_values_
            k += 1
            print('(%d/%d) ...' % (k, n))
            tmppca.fit(dx_x_norm[filter_order] / count ** 0.5) # 5
            pca['dx_norm'] = tmppca.singular_values_
            k += 1
            print('(%d/%d) ...' % (k, n))
            tmppca.fit(np.append(tar_x_norm[filter_order],dx_x_norm[filter_order],
                                 axis = 0) / count ** 0.5) # 6
            pca['all_res'] = tmppca.singular_values_
            k += 1
            print('(%d/%d) ...' % (k, n))
            print('filter (%d/%d)\tsample (%d)\t' %(filter_order + 1, filter_num, num)+'PCA Finished')
    
            # painting figures
            fig = 0
            for key in pca:
                # decent singular_values_
                plt.figure(fig)
                fig += 1
                plt.plot(pca[key], 'k', linewidth=2)
                plt.xlabel('n_components', fontsize=10)
                plt.ylabel('singular_values_', fontsize=10)
                plt.title('filter(%d/%d) ' % (filter_order + 1, filter_num) + key + ' (%d) ' % (num), fontsize=12)
                dir = './'+ name + '/decent/' + key + '/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                plt.savefig(dir + 'filter(%d,%d)' % (filter_order + 1, filter_num) + key + '(%d)decent' % (num) + '.png')
    
                # histogram singular_values_
                plt.figure(fig)
                fig += 1
                max_pca = max(pca[key])
                min_pca = min(pca[key])
                plt.hist(pca[key],np.arange(min_pca,max_pca,(max_pca - min_pca)/interval))
                plt.ylabel('number_of_components', fontsize=10)
                plt.xlabel('singular_values_', fontsize=10)
                plt.title('filter (%d/%d) ' % (filter_order + 1, filter_num) + key + ' (%d) ' % (num), fontsize=12)
                dir = './'+ name + '/hist/' + key + '/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                plt.savefig(dir + 'filter(%d,%d)' % (filter_order + 1, filter_num) + key + '(%d)hist' % (num) + '.png')
    
                # histogram singular_values_ tar without 0
                plt.figure(fig)
                fig += 1
                max_pca = max(pca[key])
                min_pca = min(pca[key])
                plt.hist(pca[key],np.arange((max_pca - min_pca)/interval,max_pca,(max_pca - min_pca)/interval))
                plt.ylabel('number_of_components', fontsize=10)
                plt.xlabel('singular_values_', fontsize=10)
                plt.title('filter (%d/%d) ' % (filter_order + 1, filter_num) + key + ' (%d) no_0 ' % (num), fontsize=12)
                dir = './'+ name + '/hist_no_0/' + key + '/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                plt.savefig(dir + 'filter(%d,%d)' % (filter_order + 1, filter_num) + key + '(%d)hist_no_0' % (num) + '.png')
    
                # bdscan
                print('DBSCAN ...')
                dbscan_data = DBSCAN(eps=0.001 * (pca[key].max() - pca[key].min()), min_samples=2).fit(
                    pca[key].reshape((-1, 1)))
                _, counts = np.unique(dbscan_data.labels_, return_counts=True)
                effective_dimension = sum(counts) - max(counts)
                print(key, 'effective dimension:', effective_dimension)
                print('DBSCAN finished!')
    
                # yx_line singular_values_ tar
                plt.figure(fig)
                fig += 1
                plt.axis([min(pca[key]), max(pca[key]), min(pca[key]), max(pca[key])])
                plt.scatter(pca[key], pca[key], marker='x', c='b')
                plt.scatter(pca[key][0:effective_dimension], pca[key][0:effective_dimension], marker='x', c='r')
                plt.ylabel('singular_values_', fontsize=10)
                plt.xlabel('singular_values_', fontsize=10)
                plt.title('filter (%d/%d) ' % (filter_order + 1, filter_num) + key + ' (%d) no_0 ' % (num) +
                          ' \n effective_dimension: %d' % effective_dimension, fontsize=12)
                dir = './'+ name + '/yx_line/' + key + '/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                plt.savefig(dir + 'filter(%d,%d)' % (filter_order + 1, filter_num) + key + '(%d)yx_line' % (num) + '.png')
    
            for i in range(fig):
                plt.figure(i).clear()
                # plt.show()
    
            # print('Saving PCA...')
            # dir = './pca_data/'+ name + '/filter(%d,%d)/' % (filter_order + 1, filter_num)
            # if not os.path.exists(dir):
            #     os.makedirs(dir)
            # torch.save(pca,dir + 'pca_pretrain_filter(%d,%d).pth.tar' % (filter_order + 1, filter_num))
            # print('PCA saved.\n')

if __name__ == '__main__':
    main()