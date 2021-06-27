# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:19:24 2019

@author: Junfeng Hu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:14:28 2019

@author: wushaowu
"""

import torch
from torchvision import datasets,transforms
import torchvision
from torch.autograd import  Variable
import numpy as np
import matplotlib.pyplot as plt

##下载minist数据集：
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])


data_train=datasets.MNIST(root="minist/",  transform=transform, train=True,
                          download=True
                          )
data_test=datasets.MNIST(root="minist/", transform=transform, train=False)

"""将训练集划分为训练集和验证集"""
data_train, data_val = torch.utils.data.random_split(data_train, [50000, 10000])

##加载：
data_loader_train=torch.utils.data.DataLoader(dataset=data_train,
                                              batch_size=64,
                                              shuffle=True)
data_loader_val=torch.utils.data.DataLoader(dataset=data_val,
                                              batch_size=64,
                                              shuffle=True)
data_loader_test=torch.utils.data.DataLoader(dataset=data_test,
                                             batch_size=64,
                                             shuffle=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(28*28,500),
                                        torch.nn.ReLU(),
                                        #torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(500,10)
                                        )
    def forward(self,x):
        #x = self.conv1(x)
        x = x.view(-1,1*28*28)
        x = self.dense(x)
        return x
model = Model()
try:
    if torch.cuda.is_available():
        model.cuda()#将所有的模型参数移动到GPU上
except:
    pass
cost = torch.nn.CrossEntropyLoss()
optimzer = torch.optim.Adam(model.parameters())

n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch{}/{}".format(epoch,n_epochs))
    print("-"*10)
    for data in data_loader_train:
        #print("train ing")
        X_train,y_train = data
        
        
        #有GPU加下面这行，没有不用加
        #X_train,y_train = X_train.cuda(),y_train.cuda()
        X_train,y_train = Variable(X_train),Variable(y_train)
        #print(X_train.size(),y_train.size())
        
        outputs = model(X_train)
        _,pred = torch.max(outputs.data,1)
        optimzer.zero_grad()
        loss = cost(outputs,y_train)
        
        loss.backward()
        optimzer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    for data in data_loader_test:
        X_test,y_test = data
        #有GPU加下面这行，没有不用加
        #X_test,y_test = X_test.cuda(),y_test.cuda()
        X_test,y_test = Variable(X_test),Variable(y_test)
        outputs = model(X_test)
        _,pred = torch.max(outputs,1)
        testing_correct += torch.sum(pred == y_test.data)
  #  print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}".format(\
  #        running_loss/len(data_train),100*running_correct/len(data_train),\
  #        100*testing_correct/len(data_test)))

    valing_correct = 0
    for data in data_loader_val:
        X_val,y_val = data
        #有GPU加下面这行，没有不用加
        #X_test,y_test = X_test.cuda(),y_test.cuda()
        X_val,y_val = Variable(X_val),Variable(y_val)
        outputs = model(X_val)
        _,pred = torch.max(outputs,1)
        valing_correct += torch.sum(pred == y_val.data)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,\
          val Accuracy is:{:.4f},\
          Test Accuracy is:{:.4f}".format(\
          running_loss/len(data_train),\
          100*running_correct/len(data_train),\
          100*valing_correct/len(data_val),\
          100*testing_correct/len(data_test)))

