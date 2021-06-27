# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:57:09 2019

@author: Junfeng Hu
"""

import torch.nn as nn  #具有共同层和成本函数的神经网络库
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
model_path = './model_pth/vgg16_bn-6c64b313.pth'
BATCH_SIZE = 1
LR = 0.01
EPOCH = 1
'''
torch.manual_seed(1)

EPOCH = 30
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
'''

training_data = torchvision.datasets.MNIST(
        root='minist/',  #dataset存储路径
        train = True , #True表示是train训练集，Flase表示测试集
        transform = torchvision.transforms.ToTensor(),  #将原数据规范化到（0,1）之间
        download=True,
        )

#打印MNIST数据集的训练集及测试集的尺寸
#print(training_data.train_data.size())
#print(training_data.train_labels.size())
# torch.Size([60000,28,28])
# torch.Size([60000])

#plt.imshow(training_data.train_data[0].numpy(),cmap='gray')  #热图
#plt.title('%f'%training_data.train_labels[0])
#plt.show()

#通过torchvision.datasets获取dataset格式可直接可置于DataLoader
train_loader = Data.DataLoader(dataset=training_data,batch_size=BATCH_SIZE,
                               shuffle=True)

# 获取测试集dataset

test_data = torchvision.datasets.MNIST(
        root = 'minist/', #dataset存储路径
        train = False, #True表示是train训练集，False表示test测试集
        transform = torchvision.transforms.ToTensor(),
        download = True,
        )
#取前全部1000个测试集样本
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
## (~, 28, 28) to (~, 1, 28, 28), in range(0,1)
test_y = test_data.test_labels
def load_data(dataset_path):
    #img = cv2.imread(dataset_path, cv2.IMREAD_GRAYSCALE)/256
    img = Image.open(dataset_path)
    img=img.convert('L')
    
    #print(img.shape)
    #img_ndarray = numpy.asarray(img, dtype='float64')/256
    transform = torchvision.transforms.ToTensor()
    #return transform(img)
    return transform(img.resize((28,28)))#.resize(26,26)
  

#vgg16_pretrained = models.vgg16(pretrained=True)
'''
class VGG16_conv(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16_conv, self).__init__()
        # VGG16 (using return_indices=True on the MaxPool2d layers)
        self.features = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv4
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv5
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True))
        self.feature_outputs = [0]*len(self.features)
        self.pool_indices = dict()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),  # 224x244 image pooled down to 7x7 from features
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, n_classes))
        
        self._initialize_weights()


    def _initialize_weights(self):
        # initializing weights using ImageNet-trained model from PyTorch
        for i, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, torch.nn.Conv2d):
                self.features[i].weight.data = layer.weight.data
                self.features[i].bias.data = layer.bias.data

    def get_conv_layer_indices(self):
        return [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    def forward_features(self, x):
        output = x
        for i, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                output, indices = layer(output)
                self.feature_outputs[i] = output
                self.pool_indices[i] = indices
                
            else:
                output = layer(output)
                self.feature_outputs[i] = output
        return output
'''  
def make_vgg_block(in_channel, out_channel, convs, pool=True):
    net = []
 
     # 不改变图片尺寸卷积
    net.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
    net.append(nn.BatchNorm2d(out_channel))
    net.append(nn.ReLU(inplace=True))

    for i in range(convs - 1):
        # 不改变图片尺寸卷积
        net.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(nn.ReLU(inplace=True))

    if pool:
        # 2*2最大池化，图片变为w/2 * h/2
        net.append(nn.MaxPool2d(2))

    return nn.Sequential(*net)


# 定义网络模型
class VGG19Net(nn.Module):
    def __init__(self):
        super(VGG19Net, self).__init__()

        net = []

        # 输入32*32，输出16*16
        net.append(make_vgg_block(3, 64, 2))

        # 输出8*8
        net.append(make_vgg_block(64, 128, 2))

        # 输出4*4
        net.append(make_vgg_block(128, 256, 4))

        # 输出2*2
        net.append(make_vgg_block(256, 512, 4))

        # 无池化层，输出保持2*2
        net.append(make_vgg_block(512, 512, 4, False))

        self.cnn = nn.Sequential(*net)

        self.fc = nn.Sequential(
            # 512个feature，每个feature 2*2
            nn.Linear(512*2*2, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
'''
Loss_list = []
Accuracy_list = [] 
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  
loss_function = nn.CrossEntropyLoss()
'''
#losses = []
#acces = []

for epoch in range(EPOCH):  
    train_loss = 0
    train_acc = 0
    start = time.time() 
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        output = net(b_x)
        ed
        loss = loss_function(output, b_y)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item()
        if step % 100 == 0:  
            print('Epoch:', epoch, '|Step:', step,  
                  '|train loss:%.4f'%loss.data.item())
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
'''
import matplotlib.pyplot as plt
plt.plot(np.arange(len(losses)), losses,color = 'r')
plt.annotate('train loss',xy=(20,losses[20]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
plt.plot(np.arange(len(acces)), acces,color = 'b')
plt.annotate('train acc',xy=(20,acces[20]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
'''