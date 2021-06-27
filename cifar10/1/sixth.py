# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:10:38 2019

@author: Junfeng Hu
"""

import numpy as np
import matplotlib.pyplot as plt
def Relu(x):
    if x>=0:
        return x
    else:
        return 0
def function(w,b,x): return w*x+b
f0 = []
f1 = []
f2 = []
x = list(np.linspace(0,1,100))
for k in x:
    f0.append(Relu(function(2,1,k)))
    f1.append(Relu(function(5,8,k)))
    f2.append(Relu(function(8,11,k)))
plt.plot(x,f0) 
plt.plot(x,f1)  
plt.plot(x,f2) 