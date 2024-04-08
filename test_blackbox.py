import os
from random import random
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import shutil
import setproctitle

import models
from autoattack import AutoAttack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--data', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'svhn'))
    parser.add_argument('--data_path', type=str)

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)

    parser.add_argument('--attack', type=str, default='L2', choices=('Linf', 'L2'))
    parser.add_argument('--eps', type=float, default=128.0)


    args = parser.parse_args()

    args.eps = args.eps / 255

    setproctitle.setproctitle(args.save_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    trainTransform = transforms.Compose([
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': 0, 'pin_memory': True} 

    if args.data == 'cifar10':
        trainDataset = dset.CIFAR10(root=args.data_path, train=True, download=True, transform=trainTransform)
        testDataset = dset.CIFAR10(root=args.data_path, train=False, download=True, transform=testTransform)
    elif args.data == 'cifar100':
        trainDataset = dset.CIFAR100(root=args.data_path, train=True, download=True, transform=trainTransform)
        testDataset = dset.CIFAR100(root=args.data_path, train=False, download=True, transform=testTransform)
    elif args.data == 'svhn':
        trainDataset = dset.SVHN(root=args.data_path, split='train', download=True, transform=trainTransform)
        testDataset = dset.SVHN(root=args.data_path, split='test', download=True, transform=testTransform)

    trainLoader = DataLoader(trainDataset, batch_size=args.batchSz, shuffle=False, **kwargs)
    testLoader = DataLoader(testDataset, batch_size=args.batchSz, shuffle=False, **kwargs)
   
    lossFunc = nn.CrossEntropyLoss(reduction="mean") 

    net = torch.load(args.model_path, map_location=torch.device('cuda'))

    trainF = open(os.path.join(args.save_path, 'train.csv'), 'w')
    testF = open(os.path.join(args.save_path, 'test.csv'), 'w')

    adversary = AutoAttack(net, norm=args.attack, eps=args.eps, version='standard', log_path=args.save_path+'/log.log')
    adversary.attacks_to_run = ['square']


    print('\nTrain error against square attack: ')
    test(args, net, trainLoader, testF, lossFunc, adversary)

    print('\nTest error against square attack: ')
    test(args, net, testLoader, testF, lossFunc, adversary)

    trainF.close()
    testF.close()



def test(args, net, testLoader, testF, lossFunc, adversary):
    l = [x for (x, y) in testLoader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in testLoader]
    y_test = torch.cat(l, 0)
    adversary.run_standard_evaluation(x_test, y_test, bs=args.batchSz)
    



if __name__=='__main__':
    main()
