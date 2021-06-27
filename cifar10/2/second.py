# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:01:05 2019

@author: Junfeng Hu
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from mgnet import MgNet
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./root/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./root/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image

'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(10)))
'''
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

net = MgNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(5):  # loop over the dataset multiple times
    #train_loss = 0#每个EPOCH累计训练误差
    #train_acc = 0
    running_loss = 0.0 
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #train_loss += loss.item()#每个EPOCH累计训练误差
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        #_, pred = outputs.max(1)#挑选出输出时值最大的位置
        #num_correct = (pred == labels).sum().item()#记录正确的个数
        #acc = num_correct / inputs.shape[0]#计算精确率
        #train_acc += acc
    #losses.append(train_loss / len(trainloader))
    #acces.append(train_acc / len(trainloader))
total = 0
correct = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %% \n' % (100 * correct / total))

print('Finished Training')


'''
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %% \n' % (100 * correct / total))

print('Finished Training')

#dataiter = iter(testloader)
#images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#outputs = net(images)
#_, predicted = torch.max(outputs, 1)

#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(4)))
'''
'''
correct = 0
total = 0
with torch.no_grad():
    k=0
    for data in testloader:
        k+=1
    #for data in testloader:
        images, labels = data
        outputs = net(images)
        prec11, prec51 = accuracy(outputs, labels, topk=(1, 5))
        top11.update(prec1[0], images.size(0))
        top51.update(prec5[0], images.size(0))
        #_, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()
        if k%128==0:
                print('test_top1',top11.avg.item())
                print('test_top5',top51.avg.item())

#print('Accuracy of the network on the 10000 test images: %d %%' % (
#    100 * correct / total))
                '''
'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(20):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    

plt.plot(np.arange(len(losses)), losses,color = 'r')
plt.annotate('train loss',xy=(3,losses[3]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
plt.plot(np.arange(len(acces)), acces,color = 'b')
plt.annotate('train acc',xy=(3,acces[3]),xycoords='data',xytext=(+10,+30),
             textcoords = 'offset points',fontsize=16,arrowprops=dict(arrowstyle='->'))
'''