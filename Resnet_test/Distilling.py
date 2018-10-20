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
from Distilling_Resnet import *
import argparse
import os
import random
import json
import shutil
import time
import warnings
import torch.backends.cudnn as cudnn
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
parser.add_argument('--gpu_id', default=3, type=int)
parser.add_argument('--sample_rate', default=1.0, type=float)
parser.add_argument('--num_teacher', default=5, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--train_norm', default=False, type=bool)
parser.add_argument('--train_lr', default=0.1, type=float)
parser.add_argument('--train_gamma', default=0.2, type=float)
parser.add_argument('--dataset', default='CIFAR100', type=str)
parser.add_argument('--train_epoch', default=200, type=int)
parser.add_argument('--train_epoch_step', default='[60,120,160]', type=str)
parser.add_argument('--distilling_lr',default=0.1, type=float)
parser.add_argument('--distilling_epoch', default=480, type=int)
parser.add_argument('--distilling_epoch_step', default='[160, 240, 320, 400]', type=str)
parser.add_argument('--save_epoch_step', default=50, type=int)
parser.add_argument('--tags',default='', type=str)
parser.add_argument('--prefix_s1',default='', type=str)
parser.add_argument('--prefix_s2',default='', type=str)
parser.add_argument('--num_branchBlock',default=4, type=int)
opt = parser.parse_args()
print('parsed options:', vars(opt))

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

    def forward(self, input, target):
        assert(len(input)==len(target))
        num_teacher = len(input)
        num_mat = np.prod(input[0].shape[0:2], dtype=float)
        loss = self.Frobenius_norm_square(torch.abs(input[0]-target[0]))
        for i in range(1,num_teacher):
            loss = loss + self.Frobenius_norm_square(input[i]-target[i])
        loss = loss / (num_teacher * num_mat)
        return loss

    def Frobenius_norm_square(self, input):
        return torch.pow(input,2).sum()

'''
DK_Dataset:
    Inherited from CIFAR10/100
    Args: 
        model_paths: an str list containing dirs of saved teacher models   
    
'''
class DK_Dataset_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, model_paths, root='', train=True, download=True, transform=None, gpu_id= 0):
        super(DK_Dataset_CIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)
        self.root_dir = root
        self.model_paths = model_paths
        self.num_teacher = len(model_paths)
        self.gpu_id = gpu_id
        self.load_models()

    def load_models(self):
        self.teacher_nets = []
        for i in range(self.num_teacher):
            net = resnet(depth=56, num_classes=10).cuda(gpu_id)
            params = torch.load(self.model_paths[i])
            net.load_state_dict(params['state_dict'])
            net.eval()
            self.teacher_nets.append(net)

    def __len__(self):
        return torchvision.datasets.CIFAR10.__len__(self)

    def __getitem__(self, idx):
        images, target = torchvision.datasets.CIFAR10.__getitem__(self, idx)
        output_fmap = []
        for i in range(self.num_teacher):
            tmp = self.teacher_nets[i](images.cuda(gpu_id, non_blocking=True))
            output_fmap.append(tmp.cpu())
        output_fmap = torch.stack(output_fmap)
        return images, output_fmap, target



def main():
    # gpu_id = 3
    datasets = opt.dataset
    samplerate = opt.sample_rate
    num_teacher = opt.num_teacher
    paths = ['./Overfit_model/checkpoint_Reset56_CIFAR100_samplerate_{0}_netid_{1}.pth.tar'.format(samplerate,i)
             for i in range(num_teacher)]
    train_normal = opt.train_norm
    depth = opt.depth
    resume_dir_s1 = '{0}checkpoint_ResNet{1}_KD_s1_{2}_sampleRate{3}.pth.tar'.format(opt.prefix_s1, depth, datasets,samplerate)
    resume_dir_s2 = '{0}checkpoint_ResNet{1}_KD_s2_{2}_sampleRate{3}.pth.tar'.format(opt.prefix_s2, depth, datasets,samplerate)
    best_prec1 = 0
    epoch_step = json.loads(opt.train_epoch_step) # list with epochs to drop lrN on
    distilling_epoch_step = json.loads(opt.distilling_epoch_step)
    num_classes = 10 if datasets == 'CIFAR10' else 100

    torch.manual_seed(opt.seed)

    logger_train_s1 = Logger('./logs/%s-%s-%s/%sKD_train_s1' % (opt.prefix_s1, 'resnet' + str(depth), datasets,'sampleRate'+str(samplerate)))
    # logger_val_s1 = Logger('./logs/%s-%s/KD_val_s1' % ('resnet' + str(depth), datasets))
    logger_train_s2 = Logger('./logs/%s-%s-%s/%sKD_train_s2' % (opt.prefix_s2, 'resnet'+str(depth), datasets,'sampleRate'+str(samplerate)))
    logger_val_s2 = Logger('./logs/%s-%s-%s/%sKD_val_s2' % (opt.prefix_s2, 'resnet'+str(depth), datasets,'sampleRate'+str(samplerate)))

    net = resnet_distilling(depth = 56, num_teacher = num_teacher, num_classes=num_classes, num_branchBlock = opt.num_branchBlock)
    net.cuda(gpu_id)


    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    criterion_distilling = KDLoss_plus().cuda(gpu_id)
    optimizer = SGD(net.parameters(),lr=opt.train_lr, momentum=0.9, weight_decay=0.0005)
    optimizer_distilling = SGD(net.parameters(), lr=opt.distilling_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=epoch_step, gamma=opt.train_gamma)
    scheduler_distilling = MultiStepLR(optimizer_distilling, milestones=distilling_epoch_step, gamma=0.1)
    cudnn.benchmark = True
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

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Number of parameters: {}'.format(sum(p.numel() for p in net.parameters())))


    if not train_normal:
        start_epoch = 0
        teacher_nets = load_teachers(paths, num_classes, gpu_id)
        if os.path.isfile(resume_dir_s1):
            print("=> loading checkpoint '{}'".format(resume_dir_s1))
            checkpoint = torch.load(resume_dir_s1)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer_distilling.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_dir_s1, checkpoint['epoch']))
            scheduler_distilling.last_epoch = start_epoch - 1
        else:
            print("=> no checkpoint found at '{}', start from 0...".format(resume_dir_s1))

        for epoch in range(start_epoch, opt.distilling_epoch):
            # distilling for one epoch
            scheduler_distilling.step()
            distilling(trainloader, net, criterion_distilling, optimizer_distilling, teacher_nets, epoch, logger_train_s1)
            save_epoch = (epoch+1)%opt.save_epoch_step==0
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet56',
                'state_dict': net.state_dict(),
                # 'best_prec1': best_prec1,
                'optimizer': optimizer_distilling.state_dict(),
            }, False, resume_dir_s1, save_epoch)

    else:
        net.output_teacher = False
        start_epoch = 0
        if os.path.isfile(resume_dir_s2):
            print("=> loading checkpoint '{}'".format(resume_dir_s2))
            checkpoint = torch.load(resume_dir_s2)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_dir_s2, checkpoint['epoch']))
            scheduler.last_epoch = start_epoch-1
        elif os.path.isfile(resume_dir_s1):
            print("=> loading checkpoint '{}'".format(resume_dir_s1))
            checkpoint = torch.load(resume_dir_s1)
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}), start from 0..."
                  .format(resume_dir_s1, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}', fail to train...".format(resume_dir_s2))
            exit()

        for epoch in range(start_epoch, opt.train_epoch):

            # train for one epoch
            scheduler.step()
            train(trainloader, net, criterion, optimizer, epoch, logger_train_s2)

            # evaluate on validation set
            prec1 = validate(testloader, net, criterion, epoch, logger_val_s2)

            # remember best prec@1 and save checkpoint
            save_epoch = (epoch + 1) % opt.save_epoch_step == 0
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'resnet56',
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, resume_dir_s2, save_epoch)


def load_teachers(model_paths, num_class, gpu):
    teacher_nets = []
    for i in range(len(model_paths)):
        net = resnet(depth=56, num_classes=num_class).cuda(gpu)
        print('Loading teachers from {}...'.format(model_paths[i]))
        params = torch.load(model_paths[i])
        print('best_prec1 for {0} teacher {1}'.format(params['best_prec1'],i))
        net.load_state_dict(params['state_dict'])
        net.eval()
        teacher_nets.append(net)
    return teacher_nets

def get_feature_map(teacher_nets, input):
    output = []
    for i in range(len(teacher_nets)):
        _, tmp = teacher_nets[i](input)
        output.append(tmp.data)
    return output

def distilling(train_loader, model, criterion, optimizer, teacher_nets, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(gpu_id, non_blocking=True)

        # compute output
        output, stu_fmaps = model(input)
        teacher_fmaps = get_feature_map(teacher_nets, input)
        loss = criterion(stu_fmaps, teacher_fmaps)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

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
              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))

    # set tensorboard for visualization
    log_dict = {'Loss': losses.avg}
    set_tensorboard(log_dict, epoch, logger)

def train(train_loader, model, criterion, optimizer, epoch, logger):
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
        output, _ = model(input)
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
            output, _ = model(input)
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
        shutil.copyfile(filename, filename[0:-8]+'_best.pth.tar')
    if save_epoch:
        shutil.copyfile(filename, filename[0:-8] + '_itr_{}.pth.tar'.format(state['epoch']))


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
