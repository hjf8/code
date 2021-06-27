# -*- coding: utf-8 -*-
"""

@author: Junfeng Hu
Created on Mon Dec 10 09:39:19 2018
"""

import cv2

import  numpy as np

img = cv2.imread('789.png',cv2.IMREAD_GRAYSCALE)#图在程序的工作路径，只要图片名
'''cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
cv2.IMREAD_GRAYSCALE：读入灰度图片
cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道'''
 
print(img.shape)#获得图片得宽度，高度和通道数

print(img.size)#获得图片得大小

print(img)
#cv2.imshow('img',img)
'''第一个参数显示窗口得名字
   第二个参数是要显示的图片'''
    
#cv2.waitKey(0)
A=img[0:200,0:200].copy()
print(A)
z=np.argmax(A,axis=1)
b=zeros((200,1),uint8)
b[0,0]=1
f=np.mat(A)*np.mat(z).T
'''waitKey()函数详解

 1--waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)

 2--如下所示: while(1){ if(waitKey(100)==27)break; } 在这个程序中,我们告诉OpenCv等待用户触发事件,等待时间为100ms，如果在这个时间段内, 用户按下ESC(ASCII码为27),则跳出循环,否则,则跳出循环

 3--如果设置waitKey(0),则表示程序会无限制的等待用户的按键事件 '''
