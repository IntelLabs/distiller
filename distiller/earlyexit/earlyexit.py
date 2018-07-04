import importlib
import argparse
import os
import shutil
import time
import numpy as np

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
import torch.nn.functional as F            # most non-linearities are here

import distiller

#import torchvision.models as models       # We need to import our own models with exits
#import models as models                   # Just use imagenet_extra_models.resnet-earlyexit
# use TensorBoard for output
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs')


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
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
parser.add_argument('--checkpointdir', default='', metavar='CHECKPOINTDIR', type=str,
                    help='Subdirectory to save checkpoint and best model files during training')
parser.add_argument('--lossweights', type=float, nargs='*', dest='lossweights',
                    help='List of loss weights for exits (e.g. --lossweights 0.1 0.3)')
parser.add_argument('--earlyexit', type=float, nargs='*', dest='earlyexit',
                    help='List of EarlyExit thresholds (e.g. --earlyexit 1.2 0.9)')
parser.add_argument('--earlyexitmodel', dest='earlyexitmodel', help='Specify file containing trained model WITH early-exit')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # if earlyexit, load early-exit model (truncated net at exit) on top of pretrained parameters
    if args.earlyexitmodel:
      if os.path.isfile(args.earlyexitmodel):
        earlyexitmodel = torch.load(args.earlyexitmodel)
        model.load_state_dict(earlyexitmodel['state_dict'], strict=False)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # capture threshold for early-exit training
    if args.earlyexit:
        print("=> using early-exit values of '{}'".format(args.earlyexit))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args.earlyexit)           
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.lossweights)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, args.earlyexit)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpointdir=args.checkpointdir)


def train(train_loader, model, criterion, optimizer, epoch, lossweights):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_exit0 = AverageMeter()
    top5_exit0 = AverageMeter()
    top1_exit1 = AverageMeter()
    top5_exit1 = AverageMeter()
    top1_exitN = AverageMeter()
    top5_exitN = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute outputs at all exits
        exitN, exit0, exit1 = model(input_var)
        loss = (lossweights[0] * criterion(exit0,target_var)) + (lossweights[1] * criterion(exit1,target_var)) + ((1.0 - (lossweights[0]+lossweights[1])) * criterion(exitN,target_var))

        # measure accuracy and record loss
        prec1_exit0, prec5_exit0 = accuracy(exit0.data, target, topk=(1, 5))
        prec1_exit1, prec5_exit1 = accuracy(exit1.data, target, topk=(1, 5))
        prec1_exitN, prec5_exitN = accuracy(exitN.data, target, topk=(1, 5))

        # while there are multiple exits, there is still just one loss (linear combo of exit losses - see above)
        losses.update(loss.data[0], input.size(0))

        top1_exit0.update(prec1_exit0[0], input.size(0))
        top5_exit0.update(prec5_exit0[0], input.size(0))
        top1_exit1.update(prec1_exit1[0], input.size(0))
        top5_exit1.update(prec5_exit1[0], input.size(0))
        top1_exitN.update(prec1_exitN[0], input.size(0))
        top5_exitN.update(prec5_exitN[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1_exit0 {top1_exit0.val:.3f} ({top1_exit0.avg:.3f})\t'
                  'Prec@5_exit0 {top5_exit0.val:.3f} ({top5_exit0.avg:.3f})\t'
                  'Prec@1_exit1 {top1_exit1.val:.3f} ({top1_exit1.avg:.3f})\t'
                  'Prec@5_exit1 {top5_exit1.val:.3f} ({top5_exit1.avg:.3f})\t'
                  'Prec@1_exitN {top1_exitN.val:.3f} ({top1_exitN.avg:.3f})\t'
                  'Prec@5_exitN {top5_exitN.val:.3f} ({top5_exitN.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1_exit0=top1_exit0, top5_exit0=top5_exit0,
                   top1_exit1=top1_exit1, top5_exit1=top5_exit1, top1_exitN=top1_exitN, top5_exitN=top5_exitN))
            niter = epoch*len(train_loader)+i
            writer.add_scalar('Train/Loss', losses.val, niter)
            writer.add_scalar('TrainExit0/Prec@1', top1_exit0.val, niter)
            writer.add_scalar('TrainExit0/Prec@5', top5_exit0.val, niter)
            writer.add_scalar('TrainExit1/Prec@1', top1_exit1.val, niter)
            writer.add_scalar('TrainExit1/Prec@5', top5_exit1.val, niter)
            writer.add_scalar('TrainExitN/Prec@1', top1_exitN.val, niter)
            writer.add_scalar('TrainExitN/Prec@5', top5_exitN.val, niter)

def validate(val_loader, model, criterion, earlyexit):
    batch_time = AverageMeter()
    losses_exit0 = AverageMeter()
    losses_exit1 = AverageMeter()
    losses_exitN = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    exit_0 = 0
    exit_1 = 0
    exit_N = 0

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output (exitN) and the rest of the exit outputs
        exitN, exit0, exit1 = model(input_var)
        # if we are running validate on a validation set with ground truth, we still calculate loss
        # we will also need loss anyway in test mode as we want to calculate accuracy.
        loss_exit0 = criterion(exit0, target_var)
        loss_exit1 = criterion(exit1, target_var)
        loss_exitN = criterion(exitN, target_var)

        # measure accuracy and record loss
        prec1_exit0, prec5_exit0 = accuracy(exit0.data, target, topk=(1, 5))
        prec1_exit1, prec5_exit1 = accuracy(exit1.data, target, topk=(1, 5))
        prec1_exitN, prec5_exitN = accuracy(exitN.data, target, topk=(1, 5))

        losses_exit0.update(loss_exit0.data[0], input.size(0))
        losses_exit1.update(loss_exit1.data[0], input.size(0))
        losses_exitN.update(loss_exitN.data[0], input.size(0))

        # take exit based on CrossEntropyLoss as a confidence measure (lower is more confident)
        if (loss_exit0.cpu().data.numpy()[0] < earlyexit[0]):
            # take the results from the early exit since lower than threshold
            top1.update(prec1_exit0[0], input.size(0))
            top5.update(prec5_exit0[0], input.size(0))
            exit_0 = exit_0 + 1
        elif (loss_exit1.cpu().data.numpy()[0] < earlyexit[1]):
            # or take the results from the next early exit, since lower than its threshold
            top1.update(prec1_exit1[0], input.size(0))
            top5.update(prec5_exit1[0], input.size(0))
            exit_1 = exit_1 + 1
        else:
            # skip the exits and include results from end of net
            top1.update(prec1_exitN[0], input.size(0))
            top5.update(prec5_exitN[0], input.size(0))
            exit_N = exit_N + 1


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss0 {loss0.val:.4f} ({loss0.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'LossN {lossN.val:.4f} ({lossN.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss0=losses_exit0,loss1=losses_exit1, lossN=losses_exitN,
                   top1=top1, top5=top5))
            niter = epoch*len(train_loader)+i
            writer.add_scalar('Train/Loss', losses.val, niter)
            writer.add_scalar('TrainExit0/Prec@1', top1_exit0.val, niter)
            writer.add_scalar('TrainExit0/Prec@5', top5_exit0.val, niter)
            writer.add_scalar('TrainExit1/Prec@1', top1_exit1.val, niter)
            writer.add_scalar('TrainExit1/Prec@5', top5_exit1.val, niter)
            writer.add_scalar('TrainExitN/Prec@1', top1_exitN.val, niter)
            writer.add_scalar('TrainExitN/Prec@5', top5_exitN.val, niter)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    print('Exit_0:')
    print(exit_0)
    print('Exit_1:')
    print(exit_1)
    print('Exit_N:')
    print(exit_N)
    print('Percent early exit #0:')
    print((exit_0*100.0) / (exit_0+exit_1+exit_N))
    print('Percent early exit #1:')
    print((exit_1*100.0) / (exit_0+exit_1+exit_N))

    return top1.avg


def save_checkpoint(state, is_best, filename='/checkpoint.pth.tar', checkpointdir=''):
    checkpointfile = checkpointdir + filename
    torch.save(state, checkpointfile)
    if is_best:
        bestfile = checkpointdir + "/model_best.pth.tar"
        shutil.copyfile(checkpointfile, bestfile)


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
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
