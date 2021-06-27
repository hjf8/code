# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:23:03 2019

@author: 11876
"""

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import scipy.io



#载入时间序列数据
#data = pd.read_csv('PMD20181122002FL00LJ.csv',header=None)
data=pd.read_csv('PMD20181122002FL00LJ.csv',usecols=['脉搏'], encoding = 'gb18030')

#EMD经验模态分解
decomposer = EMD(data['脉搏'])
imfs = decomposer.decompose()
#绘制分解图
plot_imfs(data['脉搏'],imfs,data.index)

#保存IMFs
arr = np.vstack((imfs,data['脉搏']))
dataframe = pd.DataFrame(arr.T)
dataframe.to_csv('D:/imf.csv',index=None,columns=None)
dataframe['和'] = (dataframe[1]+dataframe[2]+dataframe[3]+dataframe[4]+dataframe[5]+dataframe[6]+dataframe[7])/7
plt.plot(dataframe['和'][0:300])
#plt.plot(data['脉搏'])

