'''
Pytorch training codes using ResNet-50 for CIFAR dataset
Following the data preparation for CIFAR10&CIFAR100 in https://github.com/szagoruyko/wide-residual-networks
Referred by https://arxiv.org/abs/1605.07146
The ResNet Architecture uses the modification in https://github.com/bearpaw/pytorch-classification
Referred by https://arxiv.org/abs/1805.05551
The tensorboard gadgets uses https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

Ruofan Liang, 2018
'''

from Normal_Resnet import *
import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from logger import Logger
import os
from torch.nn.modules.loss import _Loss


parser = argparse.ArgumentParser(description='Knowledge_Distillation')
# Model options
parser.add_argument('--depth',default=56, type=int)
parser.add_argument('--gpu_id', default=1, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--train_lr', default=0.1, type=float)
parser.add_argument('--train_gamma', default=0.2, type=float)
parser.add_argument('--dataset', default='CIFAR100', type=str)
parser.add_argument('--train_epoch', default=200, type=int)
parser.add_argument('--train_epoch_step', default='[60,120,160]', type=str)
parser.add_argument('--save_epoch_step', default=500, type=int)
parser.add_argument('--tag',default='', type=str)
parser.add_argument('--num_gen', default=20, type=int)
parser.add_argument('--tau', default=1, type=float)
parser.add_argument('--lambd', default=0, type=float)
opt = parser.parse_args()
print('parsed options:', vars(opt))


cuda.empty_cache()
gpu_id = opt.gpu_id

'''
KDLoss_plus: 
    Args for forward: 
        input: tensor from different branches   shape: (num_teacher,num_sample, channel, w, h)
        target: tensor from different teacher(Over-fitting) Net
'''
class KDLoss_plus(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction=None):
        super(KDLoss_plus, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, tau=1):
        assert(len(input)==len(target))
        bs = input.shape[0]
        log_prob = nn.LogSoftmax()(input/tau)
        soft_tar = nn.Softmax()(target/tau)
        return -tau**2*torch.sum(log_prob*soft_tar)/bs

def main():
    # gpu_id = 3
    datasets = opt.dataset
    depth = opt.depth
    resume_dir = 'checkpoint_ResNet{0}_{1}_{2}.pth.tar'.format(depth, datasets, opt.tag)
    best_prec1 = 0
    epoch_step = json.loads(opt.train_epoch_step) # list with epochs to drop lrN on
    num_classes = 10 if datasets == 'CIFAR10' else 100

    logger_BAN = Logger('./logs/BAN_ResNet{}_{}_{}/BAN_prec1'.format(depth, datasets, opt.tag))
    stats = torch.load('../data/{}/stats.pkl'.format(datasets))
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

    trainset = torchvision.datasets.__dict__[datasets](root='../data/{}'.format(datasets), train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    testset = torchvision.datasets.__dict__[datasets](root='../data/{}'.format(datasets), train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    net = resnet(depth=depth, num_classes=num_classes)
    net.cuda(gpu_id)

    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    criterion_distilling = KDLoss_plus().cuda(gpu_id)
    optimizer = SGD(net.parameters(), lr=opt.train_lr, momentum=0.9, weight_decay=0.0005)

    scheduler = MultiStepLR(optimizer, milestones=epoch_step, gamma=opt.train_gamma)
    cudnn.benchmark = True

    start_gen = 0
    start_epoch = 0


    if os.path.isfile(resume_dir):
        print("=> loading checkpoint '{}'".format(resume_dir))
        checkpoint = torch.load(resume_dir)
        start_epoch = checkpoint['epoch']
        start_gen = checkpoint['gen']
        best_prec1 = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (gen {})(epoch {})"
              .format(resume_dir, start_gen, start_epoch))
        scheduler.last_epoch = start_epoch - 1

    torch.manual_seed(opt.seed+start_gen)

    if start_gen == 0:
        logger_train = Logger('./logs/BAN_ResNet{}_{}_{}/BAN_train_gen_{}'.format(depth, datasets,opt.tag, start_gen))
        logger_val = Logger('./logs/BAN_ResNet{}_{}_{}/BAN_val_gen_{}'.format(depth, datasets,opt.tag, start_gen))

        for epoch in range(start_epoch, opt.train_epoch):

            # train for one epoch
            scheduler.step()
            train(trainloader, net, criterion, optimizer, epoch, start_gen, logger_train)

            # evaluate on validation set
            prec1 = validate(testloader, net, criterion, epoch, logger_val)

            # remember best prec@1 and save checkpoint
            save_epoch = (epoch + 1) % opt.save_epoch_step == 0
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'train_epoch': opt.train_epoch,
                'gen': start_gen,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, resume_dir, save_epoch)
        set_tensorboard({'best_prec1':best_prec1}, 0, logger_BAN)
        start_gen += 1
        start_epoch = 0
        best_prec1 = 0
        scheduler.last_epoch = -1

    teacher_net = resnet(depth=opt.depth, num_classes=num_classes).cuda(gpu_id)

    for gen in range(start_gen, opt.num_gen):
        logger_train = Logger('./logs/BAN_ResNet{}_{}_{}/BAN_train_gen_{}'.format(depth, datasets, opt.tag, gen))
        logger_val = Logger('./logs/BAN_ResNet{}_{}_{}/BAN_val_gen_{}'.format(depth, datasets, opt.tag, gen))

        torch.manual_seed(opt.seed+gen)
        params = torch.load('BAN_ResNet{}_{}_gen_{}_{}.pth.tar'.format(
            opt.depth, opt.dataset, gen-1, opt.tag))
        teacher_net.load_state_dict(params['state_dict'])

        if start_epoch == 0:
            net.init_params(seed=opt.seed+gen)

        for epoch in range(start_epoch, opt.train_epoch):
            # train for one epoch
            scheduler.step()
            distilling(trainloader, net, criterion_distilling, criterion, optimizer, teacher_net, epoch, gen, logger_train)

            prec1 = validate(testloader, net, criterion, epoch, logger_val)

            save_epoch = (epoch + 1) % opt.save_epoch_step == 0
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'train_epoch': opt.train_epoch,
                'gen': gen,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, resume_dir, save_epoch)
        set_tensorboard({'best_prec1':best_prec1}, gen, logger_BAN)
        start_epoch = 0
        scheduler.last_epoch = -1
        best_prec1 = 0


def distilling(train_loader, model, criterion_soft,criterion_hard, optimizer, teacher_net, epoch, gen, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    teacher_net.eval()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(gpu_id, non_blocking=True)
        target = target.cuda(gpu_id, non_blocking=True)

        # compute output
        output = model(input)
        teacher_output = teacher_net(input).data
        loss = (1-opt.lambd)*criterion_soft(output, teacher_output,opt.tau) \
               + opt.lambd * criterion_hard(output, target)

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
        print('Gen: [{0}]\t'
              'Epoch: [{1}][{2}/{3}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            gen, epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5))

    # set tensorboard for visualization
    log_dict = {'Loss':losses.avg, 'top1_prec':top1.avg.item(),'top5_prec':top5.avg.item()}
    set_tensorboard(log_dict,  epoch, logger)

def train(train_loader, model, criterion, optimizer, epoch,gen,logger):
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
        print('Gen: [{0}]\t'
              'Epoch: [{1}][{2}/{3}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            gen, epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5))

    # set tensorboard for visualization
    log_dict = {'Loss':losses.avg, 'top1_prec':top1.avg.item(),'top5_prec':top5.avg.item()}
    set_tensorboard(log_dict,  epoch, logger)

def validate(val_loader, model, criterion, epoch, logger):
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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_epoch=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[0:-8]+'_best_gen_{}.pth.tar'.format(state['gen']))
    if save_epoch:
        shutil.copyfile(filename, filename[0:-8] + '_itr_{}.pth.tar'.format(state['epoch']))
    if state['epoch']==state['train_epoch']:
        shutil.copyfile(filename, 'BAN_ResNet{}_{}_gen_{}_{}.pth.tar'.format(
            opt.depth, opt.dataset, state['gen'], opt.tag))


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
