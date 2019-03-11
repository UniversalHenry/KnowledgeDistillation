import argparse
import os
import random
import shutil
import time
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from logger import Logger
from convOut_loader import convOut_Dataset

np.set_printoptions(precision=3)
# --suffix x4_L2  --epochs 2000  --lr 0.1 --gpu 2 --workers 2 --L2 --alphas '[0.01,0.01]' --epoch_step 1000
# --epochs 2000 --lr 0.1 --workers 2 --epoch_step 10000 --convOut_path ConvOut/convOut_CUB200_vgg16_bn_ft_L40_v1.pkl --sub_sampler sub_sampler_CUB200_M.npy --alphas '[0.001,0.001]' --gpu 0 --suffix Rx_alpha0.001
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--optim', default='SGD', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--dataset', default='CUB200', type=str)
parser.add_argument('--epoch_step', default=60, type=int)
parser.add_argument('--save_per_epoch', action='store_true')
parser.add_argument('--save_epoch', default=1000, type=int)
parser.add_argument('--logspace', default=0, type=int)
parser.add_argument('--sample_num', default=0, type=type)
parser.add_argument('--fine_tune', default=False, type=bool)
parser.add_argument('--decay_factor', default=0.2, type=float)
parser.add_argument('--device_ids', default='[3,0]', type=str)
parser.add_argument('--convOut_path', default='./ConvOut/convOut_CUB200_vgg16_bn_ft_L40_.pkl', # Zx: 40 Nx: 37
                    type=str)
parser.add_argument('--validate', action='store_true')
parser.add_argument('--model', default='v2', type=str)
parser.add_argument('--layers', default=3, type=int)
parser.add_argument('--fix_p', action='store_true', help='if add, then fix_p ')
parser.add_argument('--bn',action='store_false', help='if add, then no bn on X ')
parser.add_argument('--affine',action='store_true', help='if add, then bn affine is true ')
parser.add_argument('--sub_sampler', default='', type=str)
best_acc1 = 0

args = parser.parse_args()
device_ids = json.loads(args.device_ids)

print('parsed options:', vars(args))
if args.model == 'v2p':
    from linearTest_finetune_v2p import LinearTester
else:
    warnings.warn('no such model available')
def main():
    global args, best_acc1
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # modify for dataset:
    train_dataset = convOut_Dataset(args.convOut_path)
    if args.sub_sampler=='':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True)
    else:
        sub_idx = np.load(args.sub_sampler).tolist()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True,sampler=SubsetRandomSampler(sub_idx))

    # create model
    for i in range(args.layers):
        input_size = train_dataset.convOut1.shape[1:4]
        output_size = train_dataset.convOut2.shape[1:4]
        if i == 0:
            model = LinearTester(input_size,output_size, all_layers = args.layers, gpu_id= args.gpu, fix_p = args.fix_p, bn = args.bn,affine = args.affine)
        else:
            model.add_layer()

        if args.gpu is not None:
            model = model.cuda(args.gpu)
        elif args.distributed:
            model.cuda(device_ids[0])
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            args.gpu = device_ids[0]
            print('GPU used: ' + args.device_ids)
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
                model.cuda(device_ids[0])
            else:
                model = torch.nn.DataParallel(model).cuda(device_ids[0])

        # define loss function (criterion) and optimizer
        criterion = nn.MSELoss().cuda(args.gpu)

        if args.optim == 'SGD':
            optimizer = torch.optim.SGD([{'params':model.train_params}], lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam([{'params':model.train_params}], lr=args.lr)

        # logger_train = Logger('./logs/ILSVRC_sample_rate_{}{}/train'.format(args.sample_rate,args.suffix))
        # logger_val = Logger('./logs/ILSVRC_sample_rate_{}{}/val'.format(args.sample_rate, args.suffix))
        logger_train = Logger('./logs_M/{}_{}_{}'.format(args.dataset, args.suffix, 'layers'+str(i)))

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=torch.device("cuda:{}".format(args.gpu)))
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True

        logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)

        if args.validate:
            contrib = AverageMeter()
            model.eval()
            contrib_collect = torch.zeros((len(train_dataset), args.layers))
            with torch.no_grad():
                for i in range(len(train_dataset)):
                    pack = train_dataset[i]
                    input = pack['convOut1'].unsqueeze(0).cuda(args.gpu, non_blocking=True)
                    target = pack['convOut2'].unsqueeze(0).cuda(args.gpu, non_blocking=True)
                    output, output_n, output_contrib, res = model.val_linearity(input)
                    contrib_collect[i] = output_contrib
                    contrib.update(output_contrib)
                    if i % 100==0:
                        print("[{:4d}]/[{:4d}] Y0: {:4f} Y1: {:4f} Y2: {:.4f} | Avg: {:.4f} : {:.4f} : {:.4f}".format(
                            i, len(train_dataset),
                            output_contrib[0].item(),output_contrib[1].item(), output_contrib[2].item(),
                            contrib.avg[0].item(), contrib.avg[1].item(), contrib.avg[2].item()
                        ))
                torch.save(contrib_collect, 'contrib_collect_{}.pkl'.format(args.suffix))


        else:
            for epoch in range(args.start_epoch, args.epochs):
                if args.logspace != 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = logspace_lr[epoch]
                else:
                    adjust_learning_rate(optimizer, epoch)

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, logger_train)

                # remember best acc@1 and save checkpoint

                save_dir = 'checkpoint_{}_{}.pth.tar'.format(args.dataset, args.suffix)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, False, save_dir)
                if args.save_per_epoch and epoch > 0 and epoch % args.save_epoch == 0:
                    save_dir_itr = 'checkpoint_{}_{}_ep{}.pth.tar'.format(args.dataset, args.suffix, epoch)
                    shutil.copyfile(save_dir, save_dir_itr)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ps = AverageMeter()
    mse = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = batch['convOut1'].cuda(args.gpu, non_blocking=True)
            target = batch['convOut2'].cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        mse.update(loss.item(), input.size(0))
        # loss = loss + torch.sum((model.nonLinearLayers_p**2) * alphas)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if model.layers != 1:
            if not args.fix_p:
                ps.update(model.nonLinearLayers_p[model.layers-2].p().data.cpu().numpy())
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Layer: [{0}]\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                model.layers,
                epoch, i, len(train_loader),
                batch_time=batch_time,
                loss=losses, ))

            print('\tP {}({})'.format(ps.val, ps.avg,))

    log_dict = {'Loss': losses.avg, 'MSE': losses.avg,
                'ps': ps.avg,}
    set_tensorboard(log_dict, epoch, logger)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}_{}.pth.tar'.format(args.dataset, args.suffix))


# For tensorboard
def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, size=(1,)):
        self.size = size
        self.reset()

    def reset(self):
        self.val = 0

        if self.size != (1,):
            self.sum = np.zeros(self.size)
            self.avg = np.zeros(self.size)
        else:
            self.sum = 0
            self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.decay_factor ** (epoch // args.epoch_step))
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


if __name__ == '__main__':
    main()