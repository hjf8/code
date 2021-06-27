# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:04:24 2019

@author: Sun
"""
import torch
import numpy as np
import torch.nn as nn
import  torch.nn.functional as F


class MgIte(nn.Module):
    def __init__(self, A, B):
        super(MgIte, self).__init__()
        self.A = A
        self.B = B

    def forward(self, out):
        u, f = out
        u = u + self.B(f-self.A(u))
        out = (u, f)
        return out


class MgRestriction(nn.Module):
    def __init__(self, A1, A2, Pi, R, Bn_relu, Bn_relu_A):
        super(MgRestriction, self).__init__()
        self.A1 = A1
        self.A2 = A2
        self.Pi = Pi
        self.R = R
        self.Bn_relu = Bn_relu
        self.Bn_relu_A = Bn_relu_A

    def forward(self, out):
        u, f = out
        u1 = self.Pi(u)
        # f = self.R(f-self.A1(u)) + self.A2(u1)
        # f = self.Bn_relu(f)
        f = self.Bn_relu(self.R(f-self.A1(u))) + self.A2(u1)
        # f = self.Bn_relu(self.R(f-self.A1(u))) + self.Bn_relu_A(self.A2(u1))
        return u1, f


class MgNet(nn.Module):
    def __init__(self, features):
        super(MgNet, self).__init__()
        input_channel = 1
        num_channel_f = 64
        self.conv1 = nn.Conv2d(input_channel, num_channel_f, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel_f)
        self.features = features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128 ,10)
       

   
    
    def forward(self, u):
        f = F.relu(self.bn1(self.conv1(u)))
        shape =  np.array(f.shape)
        shape[1] = self.num_channel_u
        if self.args.cuda:
            torch_device = torch.device("cuda")
            u = torch.zeros(tuple(shape), device=torch_device)
        else:
            u = torch.zeros(tuple(shape))
        out = (u, f)
        out = self.layers(out)
        u, f = out
        u = self.pooling(u)
        u = u.view(u.shape[0], -1)
        u = self.fc(u)
        return u

cfg = [128, 256, 512]
blocks = [1,1]
def make_layers(blocks, cfg):
    layers = []
    in_channels = 64
    for i, ite_l in enumerate(blocks):
        out_channels = cfg[i]
        A2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        for ite in range(ite_l):
            B_cv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
            B = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(),
                              B_cv, nn.BatchNorm2d(in_channels), nn.ReLU())
            layers.append(MgIte(A2, B))
        out_channels = cfg[i]
        if i < len(blocks)-1:
                A1 = A2
                A2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                Pi_cv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
                Pi = nn.Sequential(Pi_cv, nn.BatchNorm2d(out_channels), nn.ReLU())
                R_cv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
                R = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(), R_cv)
                bn_relu = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU())
                layers.append(MgRestriction(A1, A2, Pi, R, bn_relu))
        in_channels = out_channels
    return nn.Sequential(*layers) # 返回一个包含了网络结构的时序容器
def mgnet11(**kwargs):
    model = MgNet(make_layers(blocks, cfg), **kwargs)
    #model.load_state_dict(torch.load(model_path))
    return model