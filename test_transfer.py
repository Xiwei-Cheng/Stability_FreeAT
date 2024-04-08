import os
from random import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import shutil
import setproctitle

import models
from PGD_Attack import L2PGDAttack, LinfPGDAttack



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--data', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'svhn'))
    parser.add_argument('--data_path', type=str)

    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--attacker_path', type=str)

    parser.add_argument('--attack', type=str, default='L2', choices=('Linf', 'L2'))
    parser.add_argument('--eps', type=float, default=128.0)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--step_size', type=float, default=32.0)
    parser.add_argument('--no_random_start', dest='random_start', action='store_false')
    parser.set_defaults(random_start=True)
    
    args = parser.parse_args()

    args.eps = args.eps / 255
    args.step_size = args.step_size / 255


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

    trainLoader = DataLoader(trainDataset, batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(testDataset, batch_size=args.batchSz, shuffle=False, **kwargs)
   
    lossFunc = nn.CrossEntropyLoss(reduction="mean") 


    trainF = open(os.path.join(args.save_path, 'train.csv'), 'w')
    testF = open(os.path.join(args.save_path, 'test.csv'), 'w')

    target_net = torch.load(args.model_path, map_location=torch.device('cuda'))
    source_net = torch.load(args.attacker_path, map_location=torch.device('cuda'))

    if args.attack == 'Linf':
        PGD_Attacker = LinfPGDAttack(source_net, eps=args.eps, alpha=args.step_size, steps=args.steps, random_start=args.random_start)
    elif args.attack == 'L2':
        PGD_Attacker = L2PGDAttack(source_net, eps=args.eps, alpha=args.step_size, steps=args.steps, random_start=args.random_start)


    print('Train error against transfer attack: ')
    test(args, target_net, trainLoader, testF, lossFunc, PGD_Attacker, source_net)

    print('Test error against transfer attack: ')
    test(args, target_net, testLoader, testF, lossFunc, PGD_Attacker, source_net)

    trainF.close()
    testF.close()



def test(args, target_net, testLoader, testF, lossFunc, attacker, source_net):
    target_net.eval()
    test_loss = 0
    nProcessed = 0
    incorrect = 0
    
    for data, target in testLoader:
        data, target = data.cuda(), target.cuda()

        max_loss = torch.zeros(data.shape[0]).cuda()
        max_perturbed_data = data.clone()

        for _ in range(args.restarts):
            if attacker == None:
                break
            
            perturbed_data = data.clone()
            perturbed_data = attacker(data, target)

            with torch.no_grad():
                output = source_net(perturbed_data)

            all_loss = F.cross_entropy(output, target, reduction='none').detach()
            max_perturbed_data[all_loss >= max_loss] = perturbed_data.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)


        with torch.no_grad():
            output = target_net(max_perturbed_data)

        lossFunc_ = nn.CrossEntropyLoss(reduction="sum") 
        test_loss += lossFunc_(output, target).item()
        pred = output.data.max(1)[1] 
        incorrect += pred.ne(target.data).cpu().sum()
        nProcessed += len(data)

    test_loss /= nProcessed 
    err = 100.*incorrect/nProcessed
    print('Average loss: {:.4f}, Error: {}/{} ({:.1f}%)\n'.format(test_loss, incorrect, nProcessed, err))

    testF.write('{},{}\n'.format(test_loss, err))
    testF.flush()

    return err



if __name__=='__main__':
    main()
