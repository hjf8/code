# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:10:15 2021

@author: Sun
"""

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
index = [0,1,2,3]
index = np.array(index)
c=['PPG信号\nECG信号', '呼吸信号\nECG信号', '心脏信号\nECG信号', '呼吸信号\n心脏信号\nECG信号']

#SBP
cnn = [13.8907, 12.5617, 13.6566, 4.6852]
mg = [10.8889, 10.4786, 10.7543, 6.0171]
ls = [14.0308, 14.0341,14.0302, 7.2524]
#DBP
cnn1 = [8.7070, 8.3088, 8.4813, 2.5340]
mg1 = [8.2641, 6.8878, 8.9236, 5.0000]
ls1 = [8.8868, 8.9380, 8.9002, 4.4607]
plt.figure()
plt.plot(cnn,color = 'blue',marker = 'o')
plt.plot(mg,color = 'red',marker = 'o')
plt.plot(ls,color = 'green',marker = 'o')
plt.legend(['CNN','MgNet','LSTM'])

plt.plot(cnn1,color = 'blue',marker = 'o',linestyle='dashed')
plt.plot(mg1,color = 'red',marker = 'o',linestyle='dashed')
plt.plot(ls1,color = 'green',marker = 'o',linestyle='dashed')
plt.xticks(index+0,c, rotation=45, fontsize=10, verticalalignment='top', fontweight='light')
plt.legend(['CNN','MgNet','LSTM'])
plt.show()
'''
index = [0,1,2,3]
index = np.array(index)
#plt.bar(index, x,0.2)
mincolor=(42/256,87/256,141/256,1/3)
midcolor=(42/256,87/256,141/256,2/3)
maxcolor=(42/256,87/256,141/256,1)
plt.bar(
        index,
        cnn,
        color=mincolor,
        width=1/4)
plt.bar(
        index+1/4,
        mg,
        color=midcolor,
        width=1/4)
plt.bar(
        index+1/2,
        ls,
        color=maxcolor,
        width=1/4)
c=['PPG信号\nECG信号', '呼吸信号\nECG信号', '心脏信号\nECG信号', '呼吸信号\n心脏信号\nECG信号']
#plt.xticks(index+1/3,c)
plt.xticks(index+1/3,c, rotation=45, fontsize=10, verticalalignment='top', fontweight='light')
plt.xlabel('数据集')
plt.ylabel('误差/mmHg')
plt.title("SBP")
plt.legend(['CNN','MgNet','LSTM'])

plt.savefig("sbp.png",dpi=1000,bbox_inches = 'tight')
plt.show()
'''
#舒张压




