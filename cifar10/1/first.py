# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:04:09 2018

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
from mgnet import MgNet

torch.manual_seed(1)

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True


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

test_loader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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
  
'''
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
'''
#cnn = CNN()  
cnn = MgNet()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  
loss_function = nn.CrossEntropyLoss()

losses = []
acces = []
for epoch in range(EPOCH):  
    train_loss = 0
    train_acc = 0
    start = time.time() 
    #top1 = AverageMeter()
    #top5 = AverageMeter()
    #top11 = AverageMeter()
    #top51 = AverageMeter()
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        output = cnn(b_x)  
        
        loss = loss_function(output, b_y)  
        #prec1, prec5 = accuracy(output, b_y, topk=(1, 5))
        #top1.update(prec1[0], b_x.size(0))
        #top5.update(prec5[0], b_x.size(0))
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item()
        if step % 100 == 0:  
            print('Epoch:', epoch, '|Step:', step,  
                  '|train loss:%.4f'%loss.data.item())
            #print('1',top1.avg.item())
            #print('5',top5.avg.item())
        #_, pred = output.max(1)#挑选出输出时值最大的位置
        #num_correct = (pred == b_y).sum().item()#记录正确的个数
        #acc = num_correct / b_x.shape[0]#计算精确率
        #train_acc += acc
    
    #duration = time.time() - start 
    #print('Training duation: %.4f'%duration)
    ##losses.append(train_loss / len(train_loader))
    #acces.append(train_acc / len(train_loader))
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            #print(data)
            cnn.eval()
            #images, labels = data
            #images, labels = images.cuda(), labels.cuda()
            outputs = cnn(images)
            #prec11, prec51 = accuracy(output, labels, topk=(1, 5))
            #top11.update(prec1[0], images.size(0))
            #top51.update(prec5[0], images.size(0))
            #if i%100==0:
            #    print('11',top11.avg.item())
            #   print('51',top51.avg.item())

            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #print(labels.size(0))
            correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
        acc = 100*correct / total
        print('%.3f'%acc)

'''
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

'''
import matplotlib.pyplot as plt
plt.plot(np.arange(len(losses)), losses,color = 'r')
plt.annotate('train loss',xy=(20,losses[20]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
plt.plot(np.arange(len(acces)), acces,color = 'b')
plt.annotate('train acc',xy=(20,acces[20]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
'''



import tensorflow as tf
a = tf.constant(2.1) #定义tensor常量
#sess=tf.Session()
b=a.eval#(session=sess)
print (b)
