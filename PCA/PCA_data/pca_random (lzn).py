from sklearn.decomposition import PCA
import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np

def main():

    filter_show = range(5)   # which filters to calculate
    interval = 100  # divide the histogram into how many parts
    # where to load the data
    print("Loading inputFeature_random ...")
    tar_data = torch.load('./conv_data/inputFeature_random_MAX.pkl')
    print("Loaded inputFeature_random !")
    print("Loading dW_random ...")
    dw_data = torch.load('./conv_data/dW_random_MAX.pkl')
    print("Loaded dW_random !")

    num, filter_num, top_num, channel, filter_col, filter_row = tar_data.shape
    count = channel * filter_row * filter_col
    tar_x = np.zeros([filter_num, num * top_num, count])
    dw_x = np.zeros([filter_num, num, count])
    dot_tar_x = np.zeros([filter_num, num * top_num, count])
    pca = {}
    for filter_order in filter_show:
        for order in range(num):
            for top_order in range(top_num):
                print("filter_order (%d/%d)\t" %(filter_order + 1,filter_num)+"order (%d/%d)\t" %(order + 1,num))
                tar_x[filter_order][order * top_num + top_order] = tar_data[order][filter_order][top_order].numpy().reshape([1,-1])
                dw_x[filter_order][order] = dw_data[order][filter_order].numpy().reshape([1,-1])
        for top_order in range(top_num):
            dot_tar_x[filter_order][top_order * num:(top_order + 1) * num] = tar_x[filter_order][top_order * num:(top_order + 1) * num] * dw_x[filter_order][:]
        tmppca={}
        print("\nfilter (%d/%d)\tdot_tar (%d)\t" % (filter_order + 1, filter_num, num) + "PCA ...")
        _, tmppca["tar"], _ = np.linalg.svd(tar_x[filter_order] / (num * top_num) ** 0.5)
        _, tmppca["dw"], _ = np.linalg.svd(dw_x[filter_order] / (num ) ** 0.5)
        _, tmppca["dot_tar"], _ = np.linalg.svd(dot_tar_x[filter_order] / (num * top_num) ** 0.5)
        tmppca["dot_res"] = tmppca["dw"] * tmppca["tar"]
        tar_x_norm = tar_x / np.mean(tar_x ** 2) ** 0.5
        dw_x_norm = dw_x / np.mean(tar_x ** 2) ** 0.5
        all_x = np.append(tar_x_norm,dw_x_norm,axis = 0)
        _, tmppca["tar_norm"], _ = np.linalg.svd(tar_x_norm[filter_order] / (num * top_num) ** 0.5)
        _, tmppca["dw_norm"], _ = np.linalg.svd(dw_x_norm[filter_order] / (num) ** 0.5)
        _, tmppca["all"], _ = np.linalg.svd(all_x / (num * (top_num + 1)) ** 0.5)
        print("filter (%d/%d)\tdot_tar (%d)\t" %(filter_order + 1, filter_num, num)+"PCA Finished\n")

        # painting figures
        fig = 0
        for key in tmppca:
            # decent singular_values_
            plt.figure(fig)
            fig += 1
            plt.plot(tmppca[key], 'k', linewidth=2)
            plt.xlabel('n_components', fontsize=10)
            plt.ylabel('singular_values_', fontsize=10)
            plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) " % (num), fontsize=12)
            plt.savefig("./res_random/decent/" + key + "/filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)decent" % (num) + ".png")

            # histogram singular_values_
            plt.figure(fig)
            fig += 1
            max_pca = max(tmppca[key])
            min_pca = min(tmppca[key])
            plt.hist(tmppca[key],np.arange(min_pca,max_pca,(max_pca - min_pca)/interval))
            plt.ylabel('number_of_components', fontsize=10)
            plt.xlabel('singular_values_', fontsize=10)
            plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) " % (num), fontsize=12)
            plt.savefig("./res_random/hist/" + key + "/filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)hist" % (num) + ".png")

            # histogram singular_values_ tar without 0
            plt.figure(fig)
            fig += 1
            max_pca = max(tmppca[key])
            min_pca = min(tmppca[key])
            plt.hist(tmppca[key],np.arange((max_pca - min_pca)/interval,max_pca,(max_pca - min_pca)/interval))
            plt.ylabel('number_of_components', fontsize=10)
            plt.xlabel('singular_values_', fontsize=10)
            plt.title("filter (%d/%d) " % (filter_order + 1, filter_num) + key + " (%d) no_0 " % (num), fontsize=12)
            plt.savefig("./res_random/hist_no_0/" + key + "/filter(%d,%d)" % (filter_order + 1, filter_num) + key + "(%d)hist_no_0" % (num) + ".png")

        for i in range(fig):
            plt.figure(i).clear()
            # plt.show()

        print("Saving PCA...")
        torch.save(tmppca,'pca_random.pth.tar')
        print("PCA saved.\n")

if __name__ == '__main__':
    main()