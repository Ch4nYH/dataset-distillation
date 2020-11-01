'''
iterative pruning for rotation or supervised task 
with lottery tickets or pretrain tickets 
support datasets: cifar10, Fashionmnist, cifar100
'''

import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from advertorch.utils import NormalizeByChannelMeanStd

from pruning_utils import *
from model import resnet18
from dataset import cifar10_dataloaders, cifar100_dataloaders, fashionmnist_dataloaders

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')

##################################### general setting #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--task', type=str, default='rotation', help='rotation or supervised classification')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='adv', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--resume', type=str, default=None, help="resume from checkpoint")


##################################### training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=160, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=1, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='80,120', help='decreasing strategy')


##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=15, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='rewind type')

best_sa = 0

def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    _, val_loader, _ = cifar10_dataloaders()
    class_number = 10
    model = resnet18(num_classes=class_number)

    # prepare dataset 
    img_results = torch.load("results.pth", map_location="cpu")

    model.cuda()
    criterion = nn.CrossEntropyLoss()

    initalization = deepcopy(model.state_dict())

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    all_result = {}
    all_result['train'] = []
    all_result['test_ta'] = []
    all_result['ta'] = []

    start_epoch = 0
    start_state = 0

    print('######################################## Start Standard Training Iterative Pruning ########################################')
    print(model.normalize)  


    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')

        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            check_sparsity(model)
            acc = train(img_results, model, criterion, optimizer, epoch)

            # evaluate on validation set
            tacc = validate(val_loader, model, criterion)
            # evaluate on test set


            all_result['train'].append(acc)
            all_result['ta'].append(tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'init_weight': initalization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        
            plt.plot(all_result['train'], label='train_acc')
            plt.plot(all_result['ta'], label='val_acc')
            plt.plot(all_result['test_ta'], label='test_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state)+'net_train.png'))
            plt.close()

        #report result
        check_sparsity(model, True)
        print('report best SA={}'.format(best_sa))

        all_result = {}
        all_result['train'] = []
        all_result['test_ta'] = []
        all_result['ta'] = []

        best_sa = 0
        start_epoch = 0

        if args.prune_type == 'pt':
            print('report loading pretrained weight')
            initalization = torch.load(os.path.join(args.save_dir, '0model_SA_best.pth.tar'), map_location = torch.device('cuda:'+str(args.gpu)))['state_dict']

        pruning_model(model, args.rate)
        check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        #rewind weight to init
        model.load_state_dict(initalization)

        #pruning using custom mask
        prune_model_custom(model, current_mask)
        check_sparsity(model)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)


def train(results, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (input, label, lr) in enumerate(results):

        image = input.cuda()
        target = label.cuda()
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr.item()

        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(results), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, label) in enumerate(val_loader):
        
        if args.task == 'rotation':
            image, target = rotation(input)
        else:
            image = input
            target = label

        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

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

def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * (int(batch / 4) + 1)), device = input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


if __name__ == '__main__':
    main()


