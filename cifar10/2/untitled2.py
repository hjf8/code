# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:33:31 2019

@author: Junfeng Hu
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
def jiecheng(n):
    if n <= 1:
        return 1
    else:
        return n*jiecheng(n-1)
def y1(x):return np.cos(2*pi*x)
def f(x,n):
    f=0
    for i in range(0,n+1):
        f =f+y1(i/n)*jiecheng(n)/(jiecheng(i)*jiecheng(n-i))*(x**(i))*(1-x)**(n-i)
    return f
y = []
f0 = []
f1 = []
f2 = []
x = list(np.linspace(0,1,100))
for k in x:
    y.append(y1(k))
    f0.append(f(k,5))    
    f1.append(f(k,50))
    f2.append(f(k,20))
plt.plot(x,y)
plt.plot(x,f0)  
plt.plot(x,f1)
plt.plot(x,f2)
plt.show()  #绘制图像
