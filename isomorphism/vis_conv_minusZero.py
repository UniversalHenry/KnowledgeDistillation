import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from logger import Logger
import os
from torch.nn.modules.loss import _Loss
import VGG_ImageNet as models
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import PIL.Image as Image
from convOut_loader import convOut_Dataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--resume', default='', type=str)
parser.add_argument('--alpha', default='', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--convout',default='./ConvOut/convOut_CUB200_vgg16_bn_ft_L40_.pkl', type=str)
parser.add_argument('--sub_sampler', default='', type=str)
parser.add_argument('--model', default='sigmoid_p', type=str)
parser.add_argument('--sep_bar', action='store_true')
parser.add_argument('--bn', action='store_false')
parser.add_argument('--suffix', default='', type=str)
args = parser.parse_args()

device_ids=[2,3]
arch='vgg16_bn_ft'
seed=0
batch_size=64
# resume='model_best_CUB200_vgg16_bn_ft_v0.pth.tar'
# resume2='model_best_CUB200__vgg16_bn_sd_10_ep300.pth.tar'
dataset='CUB200'
topk=[1,3]


l = 37

convOut_path = args.convout
resume = args.resume
gpu = args.gpu

save_img = True
cmap = 'jet'



normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

if dataset == 'ilsvrc':
    valdir = '/data/HaoChen/knowledge_distillation/ILSVRC2012_img_val'
    val_dataset = datasets.ImageFolder(
        valdir, T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()  # .cuda(gpu_id)

    optimizer = torch.optim.SGD(net.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
elif dataset == 'cifar10':
    transform_test = T.Compose([
        T.ToTensor(),  # [0, 256] -> [0, 1]
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],  # mean
                    std=[0.2470, 0.2435, 0.2616]),  # std
    ])

    val_dataset = datasets.CIFAR10(root='/data/HaoChen/knowledge_distillation/data/CIFAR10', train=True,
                                   download=True, transform=transform_test)

elif dataset == 'CUB':
    valdir = '/data/HaoChen/CUB_bird_binary/val/'

    normalize = T.Normalize(mean=[0.4795, 0.4822, 0.4218],
                            std=[0.2507, 0.2469, 0.2760])

    val_dataset = datasets.ImageFolder(valdir, T.Compose([
        T.Resize(224),
        T.ToTensor(),
        normalize,
    ]))

elif dataset == 'CUB200':
    valdir = '/data/HaoChen/CUB_200_2011/crop/test'
    val_dataset = datasets.ImageFolder(
        valdir, T.Compose([
#                 T.RandomResizedCrop(224),
#                 T.RandomHorizontalFlip(),
            T.Resize((224,224)),
#             T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=True)


denormalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225])
toPIL = T.Compose([denormalize, T.ToPILImage()])

if args.sub_sampler =='':
    sub_idx = range(len(val_dataset))
else:
    sub_idx = np.load(args.sub_sampler).tolist()


if args.model == 'v2':
    from linearTest_v2 import LinearTester
elif args.model == 'new':
    from linearTest import LinearTester
elif args.model == 'sigmoid_p':
    from linearTest_sigmoidP import LinearTester
elif args.model == 'sigmoid_p_old':
    from linearTest_sigmoidP_old import LinearTester
elif args.model == 'v2p':
    from linearTest_v2_sigmoidP import LinearTester
else:
    from linearTest_v0 import LinearTester

if args.suffix != '':
    args.suffix = '('+args.suffix+')'

train_dataset = convOut_Dataset(convOut_path)
input_size = train_dataset.convOut1.shape[1:4]
output_size = train_dataset.convOut2.shape[1:4]
model = LinearTester(input_size,output_size, gpu_id= gpu, bn=args.bn).cuda(gpu)
checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(gpu)))
model.load_state_dict(checkpoint['state_dict'])
del checkpoint

model.eval()
s_list = np.arange(0,len(sub_idx),len(sub_idx)//40)
show_list = np.arange(0,512,32)
ps = model.get_p().data.cpu().numpy()

for s in s_list:
    i = sub_idx[s]
    pack = train_dataset[i]
    input = pack['convOut1'].unsqueeze(0).cuda(gpu, non_blocking=True)
    target = pack['convOut2']
    output, output_n, output_contrib, output_zero = model.val_linearity(input)
    inputs = input.cpu().squeeze(0).numpy()
    output = output - output_zero
    for i in range(output_n.shape[0]):
        output_n[i] = output_n[i] - output_zero
    output = output.numpy()
    output_n = output_n.numpy()
    target = target.numpy()
    residual = np.abs(target - output)
    # output_n[1] = output_n[1] / ps[0]
    # output_n[2] = output_n[2] / ps[0] / ps[1]
    vmin1 = np.min((inputs, output))
    vmin2 = np.min(output_n)
    vmax1 = np.max((inputs, output))
    vmax2 = np.max(output_n)
    if not args.sep_bar:
        vmax1 = vmax2 = np.max((vmax1, vmax2))
        vmin1 = vmin2 = np.min((vmin1, vmin2))
    #############################################################

    # show_list = np.random.permutation(np.arange(512))[:32]


    img_path = 'v2p_figs/minusZero/alpha' + args.alpha +args.suffix

    sub_img_path = img_path + '/%04d'%(i)

    if save_img:
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if not os.path.exists(sub_img_path):
            os.makedirs(sub_img_path)

    sum_output = np.sum(output, 0)
    sum_target = np.sum(target, 0)
    sum_output_n = np.sum(output_n, 1)
    sum_inputs = np.sum(inputs, 0)

    svmin1 = np.min((sum_inputs, sum_output))
    svmax1 = np.max((sum_inputs, sum_output))
    svmin2 = np.min(sum_output_n)
    svmax2 = np.max(sum_output_n)
    if not args.sep_bar:
        svmax1 = svmax2 = np.max((svmax1, svmax2))
        svmin1 = svmin2 = np.min((svmin1, svmin2))

    fig1, axes = plt.subplots(2, 4, figsize=(12, 12))
    fig1.suptitle("Sum of channels Alpha {}{}".format(args.alpha,args.suffix))
    axes[0][0].imshow(toPIL(val_dataset[i][0]))
    #     axes[1][0].imshow(toPIL(val_dataset[i][0]))
    axes[0][1].imshow(sum_inputs, cmap=cmap, vmin=svmin1, vmax=svmax1)
    axes[0][1].set_title("In")
    axes[0][2].imshow(sum_target, cmap=cmap, vmin=svmin1, vmax=svmax1)
    axes[0][2].set_title("Tar")
    im0 = axes[0][3].imshow(sum_output, cmap=cmap, vmin=svmin1, vmax=svmax1)
    axes[0][3].set_title("Out")

    axes[1][0].bar([0, 1, 2], [1.0, *ps[::-1]], 0.8)
    ylim = axes[1][0].get_ylim()
    xlim = axes[1][0].get_xlim()
    axes[1][0].set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 1.5)
    axes[1][0].set_title('Ps')

    for k in range(1, 4):
        im1 = axes[1][k].imshow(sum_output_n[k - 1], cmap=cmap, vmin=svmin2, vmax=svmax2)
        axes[1][k].set_title("Y{}".format(k - 1))

    fig1.tight_layout()
    fig1.subplots_adjust(right=0.8)
    if args.sep_bar:
        cbar_ax1 = fig1.add_axes([0.85, 0.55, 0.05, 0.35])
        fig1.colorbar(im0, cax=cbar_ax1)
        cbar_ax2 = fig1.add_axes([0.85, 0.10, 0.05, 0.35])
        fig1.colorbar(im1, cax=cbar_ax2)
    else:
        cbar_ax2 = fig1.add_axes([0.85, 0.10, 0.05, 0.80])
        fig1.colorbar(im1, cax=cbar_ax2)
    plt.savefig(sub_img_path+'/'+'Sum{:04d}'.format(i))
    plt.close()

    for t in show_list[:]:
        print("Channel: ", t)
        fig1, axes = plt.subplots(2, 4, figsize=(12, 12))
        fig1.suptitle("Channel: {:03d} Alpha {}{}".format(t, args.alpha,args.suffix))
        axes[0][0].imshow(toPIL(val_dataset[i][0]))
        #     axes[1][0].imshow(toPIL(val_dataset[i][0]))
        axes[0][1].imshow(inputs[t], cmap=cmap, vmin=vmin1, vmax=vmax1)
        axes[0][1].set_title("In")
        axes[0][2].imshow(target[t], cmap=cmap, vmin=vmin1, vmax=vmax1)
        axes[0][2].set_title("Tar")
        im0 = axes[0][3].imshow(output[t], cmap=cmap, vmin=vmin1, vmax=vmax1)
        axes[0][3].set_title("Out")

        axes[1][0].bar([0, 1, 2], [1.0, *ps[::-1]], 0.8)
        ylim = axes[1][0].get_ylim()
        xlim = axes[1][0].get_xlim()
        axes[1][0].set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 1.5)
        axes[1][0].set_title('Ps')

        for k in range(1, 4):
            im1 = axes[1][k].imshow(output_n[k - 1][t], cmap=cmap, vmin=vmin2, vmax=vmax2)
            axes[1][k].set_title("Y{}".format(k - 1))

        fig1.tight_layout()
        fig1.subplots_adjust(right=0.8)
        if args.sep_bar:
            cbar_ax1 = fig1.add_axes([0.85, 0.55, 0.05, 0.35])
            fig1.colorbar(im0, cax=cbar_ax1)
            cbar_ax2 = fig1.add_axes([0.85, 0.10, 0.05, 0.35])
            fig1.colorbar(im1, cax=cbar_ax2)
        else:
            cbar_ax2 = fig1.add_axes([0.85, 0.10, 0.05, 0.80])
            fig1.colorbar(im1, cax=cbar_ax2)
        fig1.colorbar(im1, cax=cbar_ax2)

        if save_img:
            plt.savefig(sub_img_path + '/' + "Channel_{:03d}".format(t))
    #     plt.show()
        plt.close()