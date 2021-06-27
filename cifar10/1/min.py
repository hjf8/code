# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:35:50 2019

@author: Junfeng Hu
"""

from __future__ import print_function
import sys, os, argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

#cuda = torch.cuda.is_available()
cuda = False

parse = argparse.ArgumentParser(description='Pytorch MNIST Example')
parse.add_argument('--batchSize', type=int, default=64, metavar='input batch size')
parse.add_argument('--testBatchSize', type=int, default=1000, metavar='input batch size for testing')
parse.add_argument('--trainSize', type=int, default=60000, metavar='input dataset size(max=60000).Default=10000')
parse.add_argument('--nEpochs', type=int, default=1, metavar='number of epochs to train')
parse.add_argument('--lr', type=float, default=0.01, metavar='Learning rate.Deafault=0.01')
parse.add_argument('--momentum', type=float, default=0.5, metavar='Default=0.5',)
parse.add_argument('--seed', type=int, default=123, metavar='Romdom Seed to use.Default=123')

opt = parse.parse_args()

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='/mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size=opt.batchSize,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='/mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size=opt.batchSize,
    shuffle=True
)
print('===>Loading data')
with open('/mnist/processed/training.pt','rb') as f:
    dataset = torch.load(f)

# reshape image to 60000*1*28*28
data = dataset[0].view(-1, 1, 28, 28)  # 60000*1*28*28
training_data = data[:5000]
print(len(training_data))
training_labels = dataset[1][:5000]
validation_data = data[5000:6000]
validation_labels = dataset[1][5000:6000]
test_data = data[6000:7000]
test_labels = dataset[6000:7000]



#通过torchvision.datasets获取dataset格式可直接可置于DataLoader
train_loader = Data.DataLoader(dataset=training_data,batch_size=BATCH_SIZE,
                               shuffle=True)
validation_loader = Data.DataLoader(dataset=validation_data,batch_size=BATCH_SIZE,
                               shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(#(1,28,28)  ##快速搭建神经网络
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2),#(16,28,28)
        # 想要con2d卷积出来的图片尺寸没有变化，padding=（kernel_size-1)/2
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2)
                     )
        self.conv2 = nn.Sequential( # (16,14,14)  
                     nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)  
                     nn.ReLU(),  
                     nn.MaxPool2d(2) # (32,7,7)  
                     )  
        self.out = nn.Linear(32*7*7, 10)
        
    def forward(self, x):  
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = x.view(x.size(0), -1) # 将（batch，32,7,7）展平为（batch，32*7*7）  
        output = self.out(x)  
        return output  
  
cnn = CNN()  

 
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  
loss_function = nn.CrossEntropyLoss()

losses = []
acces = []
for epoch in range(EPOCH):  
    train_loss = 0
    train_acc = 0
    start = time.time() 
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        output = cnn(b_x)  
        loss = loss_function(output, b_y)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item()
        if step % 100 == 0:  
            print('Epoch:', epoch, '|Step:', step,  
                  '|train loss:%.4f'%loss.data[0])  
        _, pred = output.max(1)#挑选出输出时值最大的位置
        num_correct = (pred == b_y).sum().item()#记录正确的个数
        acc = num_correct / b_x.shape[0]#计算精确率
        train_acc += acc
        
    duration = time.time() - start 
    print('Training duation: %.4f'%duration)
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
cnn = cnn.cpu()
test_output = cnn(test_x)  
pred_y = torch.max(test_output, 1)[1].data.squeeze()
s=0
for i in range(10000):
    if pred_y[i]==test_y[i]:
        s=s+1
dd=s/test_y.size(0)
    #accuracy = sum(pred_y == test_y) / test_y.size(0) 
print('Test Acc: %.4f'%dd)
import matplotlib.pyplot as plt
plt.plot(np.arange(len(losses)), losses,color = 'r')
plt.annotate('train loss',xy=(20,losses[20]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
plt.plot(np.arange(len(acces)), acces,color = 'b')
plt.annotate('train acc',xy=(20,acces[20]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
