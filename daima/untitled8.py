# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:39:19 2020

@author: Sun
"""
x1=[]
x1.append('63231323')
x1.append('53231323')
x1.append('43231323')
T=int(input('请输入一个正整数：'))
for i in range(T):
    x = input('请输入字符串：')
    k = 0
    for j in range(3):
        if x1[j] in x:
            k += 1
    print(k)
   
import math  
x = input('请输入两个正整数：')
x=x.split()
x=[int(i) for i in x]
n = x[0]
m = x[1]
a = input('请输入n个正整数ai：')
a = a.split()
a = [int(i) for i in a]
p = input('请输入n个正整数pi：')
p = p.split()
p = [int(i) for i in p]

money = 0
shengyu_day = 0 
for j in range(n):
    h = math.ceil((a[j]+shengyu_day)/m)
    shengyu_day = h*m-a[j]  
    money = money+h*p[j]
    print(money)
    
    
import numpy as np
a = np.ones(3,dtype = np.int32) 
a.itemsize#数组中每个元素的字节大小
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    