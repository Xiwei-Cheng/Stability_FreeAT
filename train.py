import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import shutil
import setproctitle
import numpy as np

import models, wideresnet
from PGD_Attack import L2PGDAttack, LinfPGDAttack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--nEpochs', type=int, default=200)
    parser.add_argument('--data', type=str, default='cifar10', choices=('cifar10', 'cifar100'))

    parser.add_argument('--method', type=str, default='vanilla', choices=('vanilla', 'fast', 'free'))
    parser.add_argument('--attack', type=str, default='L2', choices=('Linf', 'L2'))
    parser.add_argument('--eps', type=float, default=128.0)
    parser.add_argument('--model', type=str, default='res18', choices=('res18', 'wrn34'))

    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--fast_lr', type=float, default=64.0)
    parser.add_argument('--free_lr', type=float, default=128.0)
    parser.add_argument('--free_step', type=int, default=4)

    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    args.eps = args.eps / 255.0
    args.fast_lr = args.fast_lr / 255.0
    args.free_lr = args.free_lr / 255.0

    args.save_path = os.path.join('model_pth', args.save_path)
    setproctitle.setproctitle(args.save_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)


    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': 0, 'pin_memory': True} 

    if args.data == 'cifar10':
        trainDataset = dset.CIFAR10(root='cifar10', train=True, download=True, transform=trainTransform)
        testDataset = dset.CIFAR10(root='cifar10', train=False, download=True, transform=testTransform)
    elif args.data == 'cifar100':
        trainDataset = dset.CIFAR100(root='cifar100', train=True, download=True, transform=trainTransform)
        testDataset = dset.CIFAR100(root='cifar100', train=False, download=True, transform=testTransform)

    trainLoader = DataLoader(trainDataset, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(testDataset, batch_size=args.batchSz, shuffle=False, **kwargs)
   
    lossFunc = nn.CrossEntropyLoss(reduction="mean") 

    if args.data == 'cifar10' or args.data == 'svhn':
        num_classes = 10
    elif args.data == 'cifar100':
        num_classes = 100

    if args.model == 'res18':
        net = models.ResNet18(num_classes=num_classes)
    elif args.model == 'wrn34':
        net = models.WideResNet(34, num_classes=num_classes, widen_factor=10, dropRate=0.0)

    net = net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    trainF = open(os.path.join(args.save_path, 'train.csv'), 'w')
    testF = open(os.path.join(args.save_path, 'test.csv'), 'w')

    if args.attack == 'Linf':
        PGD_Attacker = LinfPGDAttack(net, eps=args.eps, alpha=args.eps/4, steps=10, random_start=True)
    elif args.attack == 'L2':
        PGD_Attacker = L2PGDAttack(net, eps=args.eps, alpha=args.eps/4, steps=10, random_start=True)
    
    for epoch in range(1, args.nEpochs + 1):
        if args.method == 'vanilla':
            train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, PGD_Attacker)
        elif args.method == 'fast':
            if args.attack == 'Linf':
                Fast_Attacker = LinfPGDAttack(net, eps=args.eps, alpha=args.fast_lr, steps=1, random_start=True)
            elif args.attack == 'L2':
                Fast_Attacker = L2PGDAttack(net, eps=args.eps, alpha=args.fast_lr, steps=1, random_start=True)
            train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, Fast_Attacker)
        elif args.method == 'free':
            train_free(args, epoch, net, trainLoader, optimizer, trainF, lossFunc)

        adjust_opt(args, optimizer, epoch)
        torch.save(net, os.path.join(args.save_path, 'latest.pth'))

        print('\nClean Train error: ')
        test(args, epoch, net, trainLoader, testF, lossFunc, None)
        print('Train error against PGD attack: ')
        test(args, epoch, net, trainLoader, testF, lossFunc, PGD_Attacker)

        print('\nClean Test error: ')
        test(args, epoch, net, testLoader, testF, lossFunc, None)
        print('Test error against PGD attack: ')
        test(args, epoch, net, testLoader, testF, lossFunc, PGD_Attacker)


    trainF.close()
    testF.close()



def train(args, epoch, net, trainLoader, optimizer, trainF, lossFunc, attacker):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()

        if attacker is not None:
            data = attacker(data, target)

        net.train()
        optimizer.zero_grad()
        output = net(data)
        loss = lossFunc(output, target)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] 
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if batch_idx % (len(trainLoader)//10) == 0: 
            print('Train Epoch: {:.2f} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        trainF.flush()



def test(args, epoch, net, testLoader, testF, lossFunc, attacker):
    net.eval()
    test_loss = 0
    nProcessed = 0
    incorrect = 0
    
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()

        if attacker is not None:
            data = attacker(data, target)

        with torch.no_grad():
            output = net(data)

        lossFunc_ = nn.CrossEntropyLoss(reduction="sum") 
        test_loss += lossFunc_(output, target).item()
        pred = output.data.max(1)[1] 
        incorrect += pred.ne(target.data).cpu().sum()
        nProcessed += len(data)

    test_loss /= nProcessed 
    err = 100.*incorrect/nProcessed
    print('Average loss: {:.4f}, Error: {}/{} ({:.1f}%)\n'.format(
        test_loss, incorrect, nProcessed, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()


    return err



def adjust_opt(args, optimizer, epoch):
    if epoch < args.nEpochs//2: lr = args.lr
    elif epoch < args.nEpochs*3//4: lr = args.lr * 0.1
    else: lr = args.lr * 0.01
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def train_free(args, epoch, net, trainLoader, optimizer, trainF, lossFunc):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.cuda(), target.cuda()


        if args.attack == 'L2':
            delta = torch.empty_like(data).normal_()
            d_flat = delta.view(data.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(data.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*args.eps
        elif args.attack == 'Linf':
            delta = torch.empty_like(data).uniform_(-args.eps, args.eps)
        
        delta = delta.cuda()
        delta.requires_grad = True

        net.train()

        for _ in range(args.free_step):
            output = net(torch.clamp(data + delta, min=0.0, max=1.0))
            loss = lossFunc(output, target)

            optimizer.zero_grad()
            loss.backward()

            delta_grad = delta.grad.detach()

            if args.attack == 'L2':
                delta_grad_norm = torch.norm(delta_grad, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, 3,32,32) + 1e-12
                delta.data = delta.data + args.free_lr * delta_grad / delta_grad_norm

                delta_norm = torch.norm(delta.data, p=2,dim=(1,2,3)).view(-1, 1,1,1).repeat(1, 3,32,32) + 1e-12
                delta.data = ~(delta_norm > args.eps) * delta.data + args.eps * delta.data * (delta_norm > args.eps) / delta_norm
            elif args.attack == 'Linf': 
                delta.data = delta.data + args.free_lr * torch.sign(delta_grad) 
                delta.data = torch.clamp(delta.data, min=-args.eps, max=args.eps)

            delta.grad.zero_()
            optimizer.step()


        nProcessed += len(data)
        pred = output.data.max(1)[1] 
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if batch_idx % (len(trainLoader)//10) == 0: 
            print('Train Epoch: {:.2f} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        trainF.flush()






if __name__=='__main__':
    main()
