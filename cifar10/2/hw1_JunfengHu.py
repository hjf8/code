# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:11:18 2019

@author: Junfeng Hu
"""

import numpy as np
import matplotlib.pyplot as plt
import math

print('first question')
r = 1.0 #圆半径
a, b = (0., 0.)#圆心坐标
theta = np.arange(0, 2*np.pi, 0.01)
x = a + r * np.cos(theta)
y = b + r * np.sin(theta)
fig = plt.figure() 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
axes = fig.add_subplot(111) 
axes.plot(x, y, color='r')
axes.axis('equal')
plt.title('单位圆')
plt.show()

print('second question')
A = np.random.rand(3,3)*10
b = np.random.rand(3,1)*10
A = np.mat(A)
b = np.mat(b)
if np.linalg.det(A) != 0:
    print('矩阵A可逆，有唯一解')
    print(A.I*b)
 
print('third question')
a = [1,4,6,9,13,16,19,28,40,100]
b = 20
if a[0]>b:
    a.insert(0,b)
if a[-1]<b:
    a.insert(len(a),b)
for i in range(len(a)-1):
    if (a[i]<b)and(a[i+1]>b):
        a.insert(i,b)
        break
print(a[:])


print('forth question') 
a=13
i=9
while i%a!=0:
    i=int(str(i)+str(9))
    if len(str(i))>20:
        print('不存在')
        break
print(i)     
print('fifth question') 
from itertools import permutations
result = []
for i in permutations('01234567',8):
    result.append(''.join(list(i)))
result = [i for i in result if (i[0]!='0') and (i[-1] not in ['0','2','4','6'])]
print('The num is :%d'%len(result))

    
print('sixth question')
def all_prime(num):
    lst = []
    if num <= 1:
        return '0 ~ %d以内没有任何素数' %num
    for i in range(2,num+1):
        for j in range(2,int(i/2)+1):
            if i%j==0:
                break
        else:
            lst.append(i)
    return lst
print(all_prime(100))

print('seventh question')        
def findstr(substr,str):
    i=1
    while i>0:
        index = str.find(substr)    
        i -= 1
    return index
a='abcdefg'
b='def'
print(findstr(b,a))
