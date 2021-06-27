# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:47:47 2019

@author: 11876
"""

import numpy as np
a=np.array([0,0,0,0,1,1,1,1,0,0,1,1,0])
m=[]
n=[]
h=[]
for i in range(len(a)-1):
    if (a[i+1]-a[i]==1):
        b=i+1
        m.append(b)
    elif a[i+1]-a[i]==-1:
        c=i
        n.append(c)
for i in range(len(m)):
    h.append(n[i]-m[i])
k=a.sum()/2


        
    
        