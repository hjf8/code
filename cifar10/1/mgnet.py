# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:40:08 2019

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
        #print(B)
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

#num-channel-u=256,  num-channel-f=256,  num-class=10
class MgNet(nn.Module):
    def __init__(self):
        super(MgNet, self).__init__()
        num_channel_u=1
        #self.args = args
        num_channel_u=256
        num_channel_f=256
        input_channel=1
        num_class=10
        blocks=[2,1]
        #num_channel_u = num_channel_u
        self.conv1 = nn.Conv2d(input_channel, num_channel_f, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel_f)
        layers=[]
        A2 = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        for i, ite_l in enumerate(blocks):
            B_cv = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=False)
            for ite in range(ite_l):
                B = nn.Sequential(nn.BatchNorm2d(num_channel_f), nn.ReLU(),
                              B_cv, nn.BatchNorm2d(num_channel_u), nn.ReLU())
                #print(B)
                layers.append(MgIte(A2, B))

            if i < len(blocks)-1:
                A1 = A2
                A2 = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
                Pi_cv = nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False)
                Pi = nn.Sequential(Pi_cv, nn.BatchNorm2d(num_channel_u), nn.ReLU())
                R_cv = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False)
                R = nn.Sequential(nn.BatchNorm2d(num_channel_f), nn.ReLU(), R_cv)
                bn_relu = nn.Sequential(nn.BatchNorm2d(num_channel_u), nn.ReLU())
                bn_relu_A = nn.Sequential(nn.BatchNorm2d(num_channel_u), nn.ReLU())
                layers.append(MgRestriction(A1, A2, Pi, R, bn_relu, bn_relu_A))

        self.layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channel_u ,num_class)

    def forward(self, u):
        f = F.relu(self.bn1(self.conv1(u)))
        shape =  np.array(f.shape)
        shape[1] = 256
        '''
        if self.cuda:
            torch_device = torch.device("cuda")
            u = torch.zeros(tuple(shape), device=torch_device)
        else:
            u = torch.zeros(tuple(shape))
        '''
        u = torch.zeros(tuple(shape)) 
        out = (u, f)
        out = self.layers(out)
        u, f = out
        u = self.pooling(u)
        u = u.view(u.shape[0], -1)
        u = self.fc(u)
        return u

print(MgNet())



