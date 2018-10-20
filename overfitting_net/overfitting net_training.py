'''
Pytorch training codes using ResNet-50 for CIFAR dataset
Following the data preparation for CIFAR10&CIFAR100 in https://github.com/szagoruyko/wide-residual-networks
Referred by https://arxiv.org/abs/1605.07146
The ResNet Architecture uses the modification in https://github.com/bearpaw/pytorch-classification
Referred by https://arxiv.org/abs/1805.05551
The tensorboard gadgets uses https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

Ruofan Liang, 2018
'''

from resnet import *
import argparse
import os
import random
import shutil
import time
import warnings
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from logger import Logger
import os



def main():
    gpu_id = 3
    datasets = 'CIFAR10'
    torch.manual_seed(gpu_id)
    # for net_id in range(1):
    #     for sample_rate in [0.1,0.2,0.4,1]:
    #         OF_net_train(gpu_id,datasets,sample_rate,net_id)
    for net_id in range(5):
        for sample_rate in [1]:
            OF_net_train(gpu_id,datasets,sample_rate,net_id)

def OF_net_train(gpu_id,datasets,sample_rate,net_id):
    depth = 32
    resume_dir = 'checkpoint_Reset{0}_{1}_samplerate_{2}_netid_{3}.pth.tar'.format(depth, datasets,sample_rate,net_id)
    best_prec1 = 0
    epoch_step = [60, 120,160] # list with epochs to drop lrN on
    num_classes = 10 if datasets == 'CIFAR10' else 100

    logger_train = Logger('./logs/%s-%s-%s-%s/train' % ('resnet'+str(depth), datasets,'sample_rate'+str(sample_rate),'net_id'+str(net_id)))
    logger_val = Logger('./logs/%s-%s-%s-%s/val' % ('resnet'+str(depth), datasets,'sample_rate'+str(sample_rate),'net_id'+str(net_id)))
    net = resnet(depth=32, num_classes=num_classes)
    net.cuda(gpu_id)
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = SGD(net.parameters(),lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=epoch_step, gamma=0.2)
    cudnn.benchmark = True

    stats = torch.load('./data/{}/stats.pkl'.format(datasets))
    transform_test = T.Compose([
            T.ToTensor(),  # [0, 256] -> [0, 1]
            T.Normalize(stats['mean'], # mean
                        stats['std']), # std
        ])
    transform_train = T.Compose([
                T.Pad(4, padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.RandomCrop(32),
                T.ToTensor(),
                T.Normalize(stats['mean'],
                            stats['std']),
            ])
    trainset = torchvision.datasets.__dict__[datasets](root='./data/{}'.format(datasets), train=True,
                                            download=True, transform=transform_train)
    # choose samples randomly according to sample rate
    trainset,label,origin_trainset = sample_choose(trainset,sample_rate)

    trainloader = DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    testset = torchvision.datasets.__dict__[datasets](root='./data/{}'.format(datasets), train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Number of parameters: {}'.format(sum(p.numel() for p in net.parameters())))

    start_epoch = 0
    # if args.resume:
    if os.path.isfile(resume_dir):
        print("=> loading checkpoint '{}'".format(resume_dir))
        checkpoint = torch.load(resume_dir)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_dir, checkpoint['epoch']))
        scheduler.last_epoch = start_epoch-1
    else:
        print("=> no checkpoint found at '{}', start from 0...".format(resume_dir))

    for epoch in range(start_epoch, 200):

        # train for one epoch
        scheduler.step()
        train(trainloader, net, criterion, optimizer, epoch, logger_train, gpu_id)

        # evaluate on validation set
        prec1 = validate(testloader, net, criterion, epoch, logger_val, gpu_id)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'train_sample_labels':label,
            'origin_train_sample':origin_trainset,
            'picked_train_sample':trainset,
        }, is_best, resume_dir)

def sample_choose(trainset,sample_rate):
    origin_trainset = trainset
    y = []
    for j in range(min(trainset.train_labels),max(trainset.train_labels)+1):
        jth_set = [i for i in range(len(trainset.train_labels)) if trainset.train_labels[i]==j]
        tmp = random.sample(jth_set, int(len(jth_set)*sample_rate))  # too slow!
        for k in range(tmp.__len__()):
            y.append(tmp[k])
    y = random.sample(y,y.__len__())
    trainset.train_labels = [trainset.train_labels[i] for i in y]
    trainset.train_data = [trainset.train_data[i] for i in y]
    return trainset,y,origin_trainset

def train(train_loader, model, criterion, optimizer, epoch, logger,gpu_id):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(gpu_id, non_blocking=True)
        target = target.cuda(gpu_id, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if (i-1) % args.print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, top1=top1, top5=top5))

    # set tensorboard for visualization
    log_dict = {'Loss':losses.avg, 'top1_prec':top1.avg.item(),'top5_prec':top5.avg.item()}
    set_tensorboard(log_dict,  epoch, logger)

def validate(val_loader, model, criterion, epoch, logger, gpu_id):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
        set_tensorboard(log_dict, epoch, logger)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# For tensorboard
def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)

    return


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


if __name__ == '__main__':
    main()