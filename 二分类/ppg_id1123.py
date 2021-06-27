# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:50:49 2018

@author: shaowu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import os
from  scipy.io import loadmat
import h5py
import glob as gb
import scipy.signal as signal
import pywt
from untitled3 import xiaobo


def xgb_model(X_train,y_train,X_test,y_test):
    '''定义xgb模型'''
    dtrain=xgb.DMatrix(X_train,y_train)
    dval=xgb.DMatrix(X_test,y_test)
    num_rounds=50000  #迭代次数
    params={'booster':'gbtree',
            'eta':0.1, #学习率
            'max_depth':5, #树的深度
            'objective':'binary:logistic',
            'eval_metric': 'auc',# 'logloss',
            'random_seed':2018 #随机种子
            }
    watchlist = [(dtrain,'train'),(dval,'val')]
    ####模型训练：
    model=xgb.train(params,dtrain,num_rounds,watchlist,verbose_eval=100,early_stopping_rounds=100)

    return model
def my_find_peaks(data,t):
    '''
    data --list or array
    t --approximate period
    '''
    # first find the first maximum point
    old_data=data.copy()
    first_point=np.argmax(data[:t])
    data=data[first_point:]
    indexs=[first_point] #第一个最大值点
    second_point=np.argmax(data[int(t/2):int(3*t/2)])+int(t/2) #第二个最大值点
    #indexs.append(second_point)
    #plt.plot(data)
    #plt.plot(second_point,data[second_point],'r*')
    '''
    new_t=second_point
    for i in range(0,len(data),new_t): #从第二个最大值点开始，寻找最大值点
        indexs.append(np.argmax(data[int(new_t/2)+i:int(3*new_t/2)+i])+int(new_t/2)+i+first_point)
        new_t=indexs[-1]
    '''
    new_t=second_point
    i=0
    while i<len(data)-int(3*new_t/2):
        indexs.append(np.argmax(data[int(new_t/2)+i:int(3*new_t/2)+i])+int(new_t/2)+i+first_point)
        #print(indexs)
        #print('=====',i)
        #new_t=indexs[-1]-indexs[-2]
        new_t=int(np.mean(np.diff(indexs)))
        #print('new_t',new_t)
        i=i+new_t
    #plt.plot(old_data) #原始图
    #plt.plot(indexs,[old_data[i] for i in indexs],'r*')
    return indexs

def find_peaks(data):
    '''寻找极值点'''
    window = signal.general_gaussian(50, p=0.1, sig=20)
    filtered = signal.fftconvolve(window, data)
    filtered = (np.average(data) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -30)
    peaks =signal.argrelmax(filtered,order = 5)
  #  plt.plot(data)
  #  plt.plot(peaks,[data[i] for i in peaks],'r*')
  #  plt.plot(filtered)
    return filtered
#-------------------read data----------------------------------------
png_path='20181121/png/PPG/' # 数据路径
csv_path='20181121/csv/PPG/' # 数据路径
userinfo_path='20181121/userinfo-20181121.csv'

def readdata(png_path,csv_path,userinfo_path):
    files=os.listdir(png_path)
    alldata=[]
    for i in files[:-1]:
        #data=loadmat(path+files[0])
        data=pd.read_csv(open(csv_path+i[:-3]+'csv'),usecols=['脉搏'])
        alldata.append([i[:-3]]+list(data['脉搏'])) #取ppg列
    alldata=pd.DataFrame(alldata)
    alldata.columns=['用户ID']+['{}'.format(i) for i in range(alldata.shape[1]-1)]

    userinfo=pd.read_csv(userinfo_path,usecols=['用户ID','备注'],encoding='gbk')
    return alldata,userinfo.fillna(0)
alldata0,userinfo0=readdata(png_path,csv_path,userinfo_path)
alldata0['用户ID']=alldata0['用户ID'].apply(lambda x:x[1:-6])
userinfo0['备注']=userinfo0['备注'].apply(lambda x:1 if 0!=x else x)
alldata0=pd.merge(alldata0,userinfo0[['用户ID','备注']],how='left',on=['用户ID']).reset_index(drop=True)


png_path='20181122/png/PPG/' # 数据路径
csv_path='20181122/csv/PPG/' # 数据路径
userinfo_path='20181122/userinfo-20181122.csv'
alldata1,userinfo1=readdata(png_path,csv_path,userinfo_path)
alldata1['用户ID']=alldata1['用户ID'].apply(lambda x:x[1:-6])
userinfo1['备注']=userinfo1['备注'].apply(lambda x:1 if 0!=x else x)
alldata1=pd.merge(alldata1,userinfo1[['用户ID','备注']],how='left',on=['用户ID']).reset_index(drop=True)


#alldata=pd.concat([alldata0,alldata1],axis=0)

#print(label.value_counts())

def feature_create(data):
    print('提取特征...')
    new_data=[] # 用于存放特征
    
    for i,row in data.iterrows(): #遍历每条样本
    
        row = xiaobo(row[1:])
    
    
        row=np.array(row[1:][row[1:]>0])
        data1=np.diff(row,1) #一阶差分
        data2=np.diff(row,2) #二阶差分
        print(data1)
        
        # 寻找极值点：
        peaks_max=my_find_peaks(row,150)
        filtered_max=find_peaks(list(row))
        
        peaks_min=my_find_peaks(-row,150)
        filtered_min=find_peaks(list(-row))
    
        peaks1=my_find_peaks(data1,150)
        peaks2=my_find_peaks(data2,150)
        
        #求平均周期：
#        peaks=[i for i in np.diff(peaks)[0] if i>40]
#        peaks1=[i for i in np.diff(peaks1)[0] if i>40]
#        peaks2=[i for i in np.diff(peaks2)[0] if i>40]           
        tm_max=np.diff(peaks_max[1:-1]).mean()
        tm_min=np.diff(peaks_min[1:-1]).mean()
        tm1=np.diff(peaks1[1:-1]).mean()
        tm2=np.diff(peaks2[1:-1]).mean()
        
        #最大值，最小值：
        M=np.mean([filtered_max[j] for j in peaks_max])
        m=np.mean([filtered_min[j] for j in peaks_min])
        # 对一个周期内的ppg数据，求统计特征：
        if len(peaks_max)>1: #该样本至少含有一个完整周期
            data3_feat1=[]
            data3_feat2=[]
            data3_feat3=[]
            data3_feat4=[]
            data3_feat_m=[]
            data3_feat_M=[]
            for k in range(1,len(peaks_max[1:-1])):
                data3=peaks_max[1:-1][k-1:k+1]
            
                data3_feat1.append(np.mean(data3))
                data3_feat2.append(np.std(data3))
                data3_feat3.append(np.var(data3))
                data3_feat4.append(np.sum(data3))
                data3_feat_m.append(np.min(data3))
                data3_feat_M.append(np.max(data3))
                  #  data3.sum(),\
        else: # 无完整周期
            data3_feat=[-99]*6
        #小波分析：
        A4,D4,D3,D2,D1=pywt.wavedec(row,'db4',level=4)
      #  plt.plot(row,'b')
      #  plt.plot(A4,'r')
      #  plt.plot(D4,'b')
      #  plt.plot(D3,'g')
      #  plt.plot(D2,'y')
        plt.plot(D1,'y')
      #  plt.plot(pywt.waverec(A4,'db4'))
    
        # 结合该条样本所有特征：
        new_data.append([row.max(),#row.min(),
        tm_max,\
                     #data1.max(),data1.min(),tm1,\
                     #data2.max(),data2.min(),tm2,\
                     M,m,\
                     np.mean(data3_feat1),np.mean(data3_feat2),np.mean(data3_feat3),\
                     ]
                    )
    return new_data
#提取特征：
tr0=pd.concat([alldata0[['用户ID','备注']],pd.DataFrame(feature_create(alldata0))],axis=1)
tr1=pd.concat([alldata1[['用户ID','备注']],pd.DataFrame(feature_create(alldata1))],axis=1)

#结合特征：
new_data=pd.concat([tr0,tr1],axis=0).reset_index(drop=True)
new_data=new_data[new_data['备注'].notnull()].reset_index(drop=True)

#准备数据：
label=new_data['备注']
new_data=new_data.drop(['用户ID','备注'],axis=1)

import xgboost as xgb
from sklearn.model_selection import train_test_split
print('划分数据集...')
train,val,train_y,val_y=train_test_split(new_data,label,test_size=0.3,\
                                             random_state=42,stratify=label)
#模型训练：
model=xgb_model(train,train_y,val,val_y)
print("model accuracy:", metrics.accuracy_score(\
    [1 if i>0.5 else 0 for i in model.predict(xgb.DMatrix(val))],val_y))
ed
#-----------------------------------------------------------------------
#---------------------------模型训练与预测-------------------------------
import xgboost as xgb
from sklearn.model_selection import train_test_split
print('划分数据集...')
train,val,train_y,val_y=train_test_split(new_data,label,test_size=0.3,\
                                             random_state=2018,stratify=label)

params = {
'objective': 'multi:softmax',
'eta': 0.1,
'max_depth': 5,
'eval_metric': 'merror',
'seed': 1,
'missing': -999,
'num_class':14,
'silent' : 1
}

xgbtrain = xgb.DMatrix(train,train_y)
xgbval = xgb.DMatrix(val,val_y)
watchlist = [ (xgbtrain,'train'), (xgbval, 'val') ]
num_rounds=10000
print('模型训练中...')
model = xgb.train(params, xgbtrain, num_rounds, watchlist,verbose_eval=1,early_stopping_rounds=100)

#------------------------下面是其他模型-----------------------------------

from sklearn.svm import LinearSVC
from sklearn import metrics
lsvc = LinearSVC(random_state=2018)
lsvc.fit(pd.DataFrame(train).fillna(-99),train_y)
score_va = lsvc._predict_proba_lr(pd.DataFrame(val).fillna(-99))
print("model accuracy:", metrics.accuracy_score(np.argmax(score_va,axis=1),val_y))

from collections import Counter
from sklearn import neighbors
clf=neighbors.KNeighborsClassifier(algorithm='auto',leaf_size=10, metric='minkowski',
metric_params=None, n_jobs=4, n_neighbors=5, p=85,
weights='uniform') ##n_jobs为进程数，p为特征数
#训练模型：
clf.fit(pd.DataFrame(train).fillna(-99),train_y)
#预测：
pre_y= clf.predict(pd.DataFrame(val).fillna(-99))
print("model accuracy:", metrics.accuracy_score(pre_y,val_y))
