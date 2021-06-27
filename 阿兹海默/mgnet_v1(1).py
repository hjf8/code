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
    def __init__(self, args):
        super(MgNet, self).__init__()
        self.args = args
        self.num_channel_u = args.num_channel_u
        self.conv1 = nn.Conv2d(args.input_channel, args.num_channel_f, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(args.num_channel_f)
        layers=[]
        A2 = nn.Conv2d(args.num_channel_u, args.num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        for i, ite_l in enumerate(args.blocks):
            B_cv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, stride=1, padding=1, bias=False)
            for ite in range(ite_l):
                B = nn.Sequential(nn.BatchNorm2d(args.num_channel_f), nn.ReLU(),
                              B_cv, nn.BatchNorm2d(args.num_channel_u), nn.ReLU())
                layers.append(MgIte(A2, B))

            if i < len(args.blocks)-1:
                A1 = A2
                A2 = nn.Conv2d(args.num_channel_u, args.num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
                Pi_cv = nn.Conv2d(args.num_channel_u, args.num_channel_u, kernel_size=3, stride=2, padding=1, bias=False)
                Pi = nn.Sequential(Pi_cv, nn.BatchNorm2d(args.num_channel_u), nn.ReLU())
                R_cv = nn.Conv2d(args.num_channel_f, args.num_channel_u, kernel_size=3, stride=2, padding=1, bias=False)
                R = nn.Sequential(nn.BatchNorm2d(args.num_channel_f), nn.ReLU(), R_cv)
                bn_relu = nn.Sequential(nn.BatchNorm2d(args.num_channel_u), nn.ReLU())
                bn_relu_A = nn.Sequential(nn.BatchNorm2d(args.num_channel_u), nn.ReLU())
                layers.append(MgRestriction(A1, A2, Pi, R, bn_relu, bn_relu_A))

        self.layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(args.num_channel_u ,args.num_class)

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

