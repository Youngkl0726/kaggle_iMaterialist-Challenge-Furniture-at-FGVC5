import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import densenet
from functools import partial

import os.path as osp
from memcached_dataset import McDataset
from distributed_utils import dist_init, average_gradients, DistModule

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=16, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float)
parser.add_argument('--decay-scale', default=0.1, type=float)
parser.add_argument('--decay-epoch', default=30, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--resume-opt', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--port', default='29046', type=str)
# parser.add_argument('--train-root', default='/mnt/lustre/yangkunlin/furniture/data/train_all/train_v1/', type=str)
# parser.add_argument('--train-source', default='/mnt/lustre/yangkunlin/furniture/data/train_all_list.txt', type=str)
parser.add_argument('--train-root', default='/mnt/lustre/yangkunlin/furniture/data/', type=str)
parser.add_argument('--train-source', default='/mnt/lustre/yangkunlin/furniture/data/train.txt', type=str)
parser.add_argument('--val-root', default='/mnt/lustre/yangkunlin/furniture/data/val/', type=str)
parser.add_argument('--val-source', default='/mnt/lustre/yangkunlin/furniture/data/valid.txt', type=str)
parser.add_argument('--save-path', default='checkpoint6', type=str)

best_prec1 = 0
min_loss = float("inf")
NB_CLASSES = 128

class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(means=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


class FinetunePretrainedModels(nn.Module):
    def __init__(self, num_classes, net_cls, net_kwards):
        super().__init__()
        self.net = net_cls(**net_kwards)
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)

    def forward(self, x):
        return self.net(x)
model_dict = {
    'dense161': partial(FinetunePretrainedModels, NB_CLASSES, densenet.densenet161)
}

net_kwards = [{'pretrained': 'True'}, {'pretrained': None}]

def get_model(model_name: str, pretrained=True):
    # print('[+] getting model architecture... ')
    if(pretrained):
        model = model_dict[model_name](net_kwards[0])
    else:
        model = model_dict[model_name](net_kwards[1])
    print('[+] done.')
    return model

last_layer_names = ['module.net.last_linear.weight', 'module.net.last_linear.bias',
                    'module.net.classifier.weight', 'module.net.classifier.bias']

def main():
    global args, best_prec1, min_loss
    args = parser.parse_args()

    rank, world_size = dist_init(args.port)
    assert (args.batch_size % world_size == 0)
    assert (args.workers % world_size == 0)
    args.batch_size = args.batch_size // world_size
    args.workers = args.workers // world_size

    # create model
    print("=> creating model '{}'".format("dense161"))
    print("save_path is: {}".format(args.save_path))
    image_size = 256
    input_size = 224
    model = get_model('dense161', pretrained=True)
    # print("model is: {}".format(model))
    model.cuda()
    model = DistModule(model)

    # optionally resume from a checkpoint
    if args.load_path:
        if args.resume_opt:
            best_prec1, start_epoch = load_state(args.load_path, model, optimizer=optimizer)
        else:
            # print('load weights from', args.load_path)
            load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = McDataset(
        args.train_root,
        args.train_source,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))
    val_dataset = McDataset(
        args.val_root,
        args.val_source,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(input_size),
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    lr = 0
    patience = 0
    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        train_sampler.set_epoch(epoch)

        if epoch == 1:
            lr = 0.00003
        if patience == 2:
        # if epoch == 1:
            patience = 0
            checkpoint = load_checkpoint(args.save_path+'_best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            print("Loading checkpoint_best.............")
            # model.load_state_dict(torch.load('checkpoint_best.pth.tar'))
            lr = lr / 10.0

        if epoch == 0:
            lr = 0.001
            for name, param in model.named_parameters():
                # print("name is: {}".format(name))
                if (name not in last_layer_names):
                    param.requires_grad = False
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        else:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=0.0001)
        print("lr is: {}".format(lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_prec1, val_losses = validate(val_loader, model, criterion)
        print("val_losses is: {}".format(val_losses))
        # remember best prec@1 and save checkpoint
        if rank == 0:
            # remember best prec@1 and save checkpoint
            if val_losses < min_loss:
                is_best = True
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'dense161',
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.save_path)
                # torch.save(model.state_dict(), 'best_val_weight.pth')
                print('val score improved from {:.5f} to {:.5f}. Saved!'.format(min_loss, val_losses))

                min_loss = val_losses
                patience = 0
            else:
                patience += 1
        if rank == 1 or rank == 2 or rank == 3 or rank == 4 or rank == 5 or rank == 6 or rank == 7:
            if val_losses < min_loss:
                min_loss = val_losses
                patience = 0
            else:
                patience += 1
        print("patience is: {}".format(patience))
        print("min_loss is: {}".format(min_loss))
    print("min_loss is: {}".format(min_loss))



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        loss = criterion(output, target_var) / world_size
        prec1 = accuracy(output.data, target, topk=(1, 1))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size

        dist.all_reduce_multigpu([reduced_loss])
        dist.all_reduce_multigpu([reduced_prec1])

        losses.update(reduced_loss[0], input.size(0))
        top1.update(reduced_prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                    data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        loss = criterion(output, target_var) / world_size
        prec1 = accuracy(output.data, target, topk=(1, 1))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size

        dist.all_reduce_multigpu([reduced_loss])
        dist.all_reduce_multigpu([reduced_prec1])

        losses.update(reduced_loss[0], input.size(0))
        top1.update(reduced_prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 and rank == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    if rank == 0:
        print('Prec@1 {top1.avg:.3f}\t'
              'Loss {loss.avg:.4f}'.format(top1=top1, loss=losses))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


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
    """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""
    if epoch < 10:
        lr = args.lr * (args.decay_scale ** (epoch // 10))
    else:
        lr = args.lr * (args.decay_scale ** (1 + ((epoch - 10) // 5)))
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
    return res[0]

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def load_state(load_path, model, optimizer=None):
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ckpt_keys = set(checkpoint['state_dict'].keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print('missing keys from checkpoint {}: {}'.format(load_path, k))

        print("=> loaded model from checkpoint '{}'".format(load_path))
        if optimizer != None:
            best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> also loaded optimizer from checkpoint '{}' (epoch {})"
                .format(load_path, start_epoch))
            return best_prec1, start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


if __name__ == '__main__':
    main()
