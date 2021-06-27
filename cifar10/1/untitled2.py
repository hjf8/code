# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:23:29 2019

@author: Junfeng Hu
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models

batch_size = 16
learning_rate = 0.0002
epoch = 10

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dir = './VGGDataSet/train'
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

val_dir = './VGGDataSet/val'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)


class VGGNet(nn.Module):
    def __init__(self, num_classes=3):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#--------------------训练过程---------------------------------
model = VGGNet()
if torch.cuda.is_available():
    model.cuda()
params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

Loss_list = []
Accuracy_list = []



for epoch in range(100):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_datasets)), train_acc / (len(train_datasets))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in val_dataloader:
        batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_datasets)), eval_acc / (len(val_datasets))))
	#Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))
ed
x1 = range(0, 100)
x2 = range(0, 100)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
# plt.savefig("accuracy_loss.jpg")
