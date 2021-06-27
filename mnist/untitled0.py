# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:13:55 2019

@author: Sun
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
# Hyper parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

#
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False
)
test_x = Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

# CNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(       #(1*28*28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1, #步长
                padding=2,
            ),    #(16*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(16*14*14)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(16,32,5,1,2),  #32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32*7*7
        )
        self.out = nn.Linear(32*7*7,10)  #全连接

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   #(batch,32,7,7)
        x = x.view(x.size(0),-1) #(batch,32*7*7)
        output = self.out(x)
        return output

cnn = CNN()
# print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()
# training and testing
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = ((pred_y == test_y).sum()/test_y.shape[0])
            print('| test accuracy: ',accuracy)#'Epoch: ',epoch,'| train loss: %4.f' %loss.data[0],

# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')