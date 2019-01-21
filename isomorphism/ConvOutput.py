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
import torchvision.models as models

parser = argparse.ArgumentParser(description='Tracing')

parser.add_argument('--device_ids', default='[0,1]', type=str)
parser.add_argument('--arch', default='vgg16_bn', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--resume1', default='model_best_CUB200__vgg16_bn_sd_1_ep300.pth.tar', type=str)
parser.add_argument('--resume2', default='model_best_CUB200__vgg16_bn_sd_10_ep300.pth.tar', type=str)
parser.add_argument('--dataset', default='CUB200', type=str)
parser.add_argument('--conv_layer', default=37, type=int)
parser.add_argument('--topk', default='[1,3]', type=str)
opt = parser.parse_args()
print('parsed options:', vars(opt))

device_ids = json.loads(opt.device_ids)
topk = json.loads(opt.topk)

if opt.dataset == 'cifar10':
    from VGG_CIFAR import *
else:
    from VGG_ImageNet import *

cuda.empty_cache()
def load_checkpoint(resume, model):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=torch.device("cuda:{}".format(device_ids[0])))
        state_dict = checkpoint['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            if key.find('module'):
                state_dict[key.replace('module.','')] = state_dict.pop(key)

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {} acc1 {})"
              .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    del checkpoint, state_dict

def main():
    cudnn.benchmark = True
    if opt.dataset == 'CUB200':
        net1 = models.__dict__[opt.arch](num_classes=200).cuda(device_ids[0])
        net2 = models.__dict__[opt.arch](num_classes=200).cuda(device_ids[1])

    load_checkpoint(opt.resume1, net1)
    load_checkpoint(opt.resume2, net2)

    net1.eval()
    net2.eval()

    # Data loading code
    # traindir = os.path.join(opt.data, 'train')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    if opt.dataset == 'ilsvrc':
        valdir = '../ILSVRC2012_img_val'
        val_dataset = datasets.ImageFolder(
            valdir, T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ]))
        val_loader = DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        criterion = nn.CrossEntropyLoss()  # .cuda(gpu_id)

        optimizer = torch.optim.SGD(net.parameters(), opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    elif opt.dataset == 'cifar10':
        transform_test = T.Compose([
            T.ToTensor(),  # [0, 256] -> [0, 1]
            T.Normalize(mean=[0.4914, 0.4822, 0.4465],  # mean
                        std=[0.2470, 0.2435, 0.2616]),  # std
        ])

        val_dataset = datasets.CIFAR10(root='../data/CIFAR10', train=True,
                                       download=True, transform=transform_test)

    elif opt.dataset == 'CUB':
        valdir = '../../CUB_bird_binary/val/'

        normalize = T.Normalize(mean=[0.4795, 0.4822, 0.4218],
                                std=[0.2507, 0.2469, 0.2760])

        val_dataset = datasets.ImageFolder(valdir, T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]))

    elif opt.dataset == 'CUB200':
        valdir = '../../CUB_200_2011/crop/test'
        val_dataset = datasets.ImageFolder(
            valdir, T.Compose([
                T.Resize((224,224)),
                # T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ]))
        val_loader = DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=6, pin_memory=True)
        # class_map = [[] for i in range(200)]
        # for i in range(len(val_dataset)):
        #     class_map[val_dataset[i][1]].append(i)

#############################
    tar = torch.zeros(len(val_dataset),dtype=torch.int)
    pred1 = torch.zeros((len(val_dataset),len(topk)), dtype=torch.int)
    convOut1 = torch.zeros((len(val_dataset),512,14,14))
    convOut2 = torch.zeros((len(val_dataset), 512, 14, 14))
    pred2 = torch.zeros((len(val_dataset),len(topk)), dtype=torch.int)

    accum_cnt = 0
    num_batches = len(val_dataset) // opt.batch_size
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = target.size(0)
            convOutBs1, predBs1 = validate(input, target, net1, device_ids[0])
            convOutBs2, predBs2 = validate(input, target, net2, device_ids[1])
            tar[accum_cnt:(accum_cnt+batch_size)] = target
            pred1[accum_cnt:(accum_cnt+batch_size),:] = predBs1
            pred2[accum_cnt:(accum_cnt+batch_size),:] = predBs2
            convOut1[accum_cnt:(accum_cnt+batch_size),:,:,:] = convOutBs1.data.cpu()
            convOut2[accum_cnt:(accum_cnt+batch_size),:,:,:] = convOutBs2.data.cpu()
            accum_cnt += batch_size
            print("Batch: {}/{}".format(i,num_batches))
        save_dict = {
            'target': tar,
            'pred1': pred1,
            'pred2': pred2,
            'convOut1': convOut1,
            'convOut2': convOut2,
        }
        torch.save(save_dict, 'convOut_{}_{}_L{}_{}.pkl'.format(opt.dataset,opt.arch,opt.conv_layer, opt.suffix))




def validate(input, target, model, device_id):
    input = input.cuda(device_id, non_blocking=True)
    target = target.cuda(device_id, non_blocking=True)
    model.eval()
    with torch.no_grad():
        convOut = model.features[:opt.conv_layer+1](input)
        featureOut = model.features[opt.conv_layer+1:](convOut)
        featureOut = featureOut.reshape(featureOut.size(0), -1)
        output = model.classifier(featureOut)
        pred = get_pred(output, target, topk)
        return convOut, pred



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_pred(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = torch.zeros((len(topk),target.size(0)),dtype=torch.int)
        for i in range(len(topk)):
            res[i,:] = torch.sum(correct[:topk[i],:],0).data.cpu()
        return res.t()


if __name__ == '__main__':
    main()


