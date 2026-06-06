#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from client.dp_mechanism import simple_composition, advanced_composition, renyi_gaussian_composition
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from client.Update import Client
from client.Nets import CNNCifar, CNNFemnist, CharLSTM, LeNet, CNNMnist
from server.FedAvg import FedAvg
from utils.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule


def fed_train(command_str):
    # 基于差分隐私的联邦学习模块，命令行的格式见options.py
    args = args_parser(command_str.split())  # parse args

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    shape_img = [32, 32]
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.Resize(shape_img), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,), )])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 3
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'LeNet' and args.dataset == 'mnist':
        net_glob = LeNet().to(args.device)  # args=args
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    total_epsilon = args.dp_epsilon
    total_delta = args.dp_delta
    dp_alpha = 4
    dp_epsilon = 0.1
    dp_delta = 1e-7

    if args.dp_composition == 'Simple':
        dp_epsilon, dp_delta = simple_composition(args.frac * args.epochs, total_epsilon, total_delta)
    elif args.dp_composition == 'Advanced':
        dp_epsilon, dp_delta = advanced_composition(args.frac * args.epochs, total_epsilon, total_delta)
    elif args.dp_composition == 'Renyi':
        dp_alpha, dp_epsilon = renyi_gaussian_composition(args.frac * args.epochs, total_epsilon, total_delta)
    # dp_epsilon = args.dp_epsilon / (args.frac * args.epochs)
    # dp_delta = args.dp_delta / (args.frac * args.epochs)
    dp_composition = args.dp_composition
    dp_mechanism = args.dp_mechanism
    dp_clip = args.dp_clip
    # use opacus to wrap model to clip per sample gradient
    net_glob = GradSampleModule(net_glob)
    # print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    rootpath = './Results/DP-FL/fed_{}_{}_{}_C{}_iid{}_dp_{}_dp_{}_epsilon_{}'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_composition,
        args.dp_epsilon)
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    # training
    acc_test = []
    learning_rate = [args.lr for i in range(args.num_users)]
    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)  # 一轮选择的客户端数量
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 随机抽取m个客户端
        # begin_index = iter % (1 / args.frac)
        # idxs_users = all_clients[int(begin_index * args.num_users * args.frac):
        #                          int((begin_index + 1) * args.num_users * args.frac)]
        for idx in idxs_users:  # 被抽取的客户端进行本地训练
            args.lr = learning_rate[idx]
            # print("before learning_rate[idx]", learning_rate[idx])
            local_client = Client(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                  dp_alpha=dp_alpha, dp_epsilon=dp_epsilon, dp_delta=dp_delta,
                                  dp_mechanism=dp_mechanism, dp_composition=dp_composition, dp_clip=dp_clip)
            w, loss, curLR = local_client.train(net=copy.deepcopy(net_glob).to(args.device))
            learning_rate[idx] = curLR
            # print("afterlearning learning_rate[idx]", learning_rate[idx])
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))
        print("Round {:3d},Testing loss: {:.2f}".format(iter, loss_t))
        acc_test.append(acc_t.item())

        # net_glob.load_state_dict(w_glob)
        state_dict = net_glob.state_dict()
        net_path = rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_dp_{}_epsilon_{}.pth'.format(
            args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_composition,
            args.dp_epsilon)
        torch.save(state_dict, net_path)


    accfile = open(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_dp_{}_epsilon_{}_acc.dat'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_composition,
        args.dp_epsilon), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_dp_{}_epsilon_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_composition,
        args.dp_epsilon))



if __name__ == '__main__':
    command_str = "--dataset mnist --model LeNet --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --dp_clip 10  --lr_decay 1 --dp_composition Renyi"
    fed_train(command_str)

