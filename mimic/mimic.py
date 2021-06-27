# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:45:22 2019

@author: shaowu
"""
import gc
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from xiaobo import xiaobo
import xlwt
import pandas as pd
#files=os.listdir('mimic/')
#files=os.listdir('mimic/')
for i in ['Part_1.mat','Part_2.mat','Part_3.mat','Part_4.mat']:
    data=h5py.File(i,'r')
    data=[data[e[0]][:][100:3120,:2] for e in data['Part_1'] if len(data[e[0]][:])>3000] #长度大于1000的序列
    gc.collect()
    res=[]
    ret=[]
    for i in range(len(data)):
        z=data[i]
        #zz=z[:,0]
        row = xiaobo(z[:,0])
        res.append(row)
        ret.append(z[:,1])
    break
np.save('ppg1.npy',res)
'''
np.save('ppg.npy',res)
ppg = pd.DataFrame(res)
ppg.to_csv('ppg.csv')
xueya = pd.DataFrame(ret)
xueya.to_csv('xueya.csv')
c=np.load('ppg.npy')
'''