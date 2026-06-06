#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w_list):
    """
    FedAvg算法
    :param w_list:client的参数组成的列表
    :return w_avg:新的全局参数
    """
    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[k] += w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_list))
    # print("len(w_list) =", len(w_list))
    return w_avg
