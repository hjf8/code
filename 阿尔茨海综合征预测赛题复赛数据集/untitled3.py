# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:38:58 2019

@author: Sun
"""

import os
import pandas as pd
import numpy as np
import time
from tqdm import *
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,recall_score,f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from collections import Counter
def one_hot_col(col):
    '''标签编码'''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl
##读取数据：
preliminary_list_test=pd.read_csv("data_final/2_final_list_test.csv") #大小（27,2）
preliminary_list_train=pd.read_csv("data_final/2_final_list_train.csv") #大小（179,2）
egemaps_pre=pd.read_csv("data_final/egemaps_final.csv") #大小（206,89）
print('训练集标签分布：\n',preliminary_list_train['label'].value_counts())

#==================================================================================
##读取转写文本，即tsv文件夹下的文件，每一个文件都是对应一个人的，且每个文件行数不一定相等。
tsv_path_lists=os.listdir('data_final/tsv2/') #大小：206
tsv_feats=[] ##用于存放tsv特征
for path in tqdm(tsv_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data_final/tsv2/'+path,sep='\t')
    ##说一句话所用时长：
    z['end_time-start_time']=z['end_time']-z['start_time']
    z['jicha'] = z['end_time-start_time'].max()-z['end_time-start_time'].min()
    tsv_feats.append([path[:-4],\
                      z['end_time-start_time'].mean(),\
                      #z['end_time-start_time'].min(),\
                      #z['end_time-start_time'].max(),\
                      #z['end_time-start_time'].std(),\
                      z['end_time-start_time'].median(),\
                      z['end_time-start_time'].skew(),\
                      z['end_time-start_time'].var(),\
                      z.shape[0]])
tsv_feats=pd.DataFrame(tsv_feats)
tsv_feats.columns=['uuid']+['tsv_feats{}'.format(i) for i in range(tsv_feats.shape[1]-1)]
#====================================================================================
##读取帧级别的Low-level descriptors (LLD)特征，即egemaps文件夹下的文件，每一个文件都是对应一个人的，且每个文件行数不一定相等。
##字段含义参考文献2
egemaps_path_lists=os.listdir('data_final/egemaps2/') #大小：206
egemaps_feats=[] ##用于存放egemaps特征
for path in tqdm(egemaps_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data_final/egemaps2/'+path,sep=';')
    z=z.drop(['name'],axis=1)
    egemaps_feats.append([path[:-4]]+\
                         list(z.mean(axis=0))+\
                         list(z.std(axis=0))+\
                         list(z.min(axis=0))+\
                         list(z.median(axis=0))) ##这里只求每列的平均值
egemaps_feats=pd.DataFrame(egemaps_feats)
egemaps_feats.columns=['uuid']+['egemaps_feats{}'.format(i) for i in range(egemaps_feats.shape[1]-1)]

#===========================分割线=============================================
##结合特征 ：
preliminary_list_train=preliminary_list_train.merge(egemaps_pre,how='left',on=['uuid'])
preliminary_list_train=preliminary_list_train.merge(tsv_feats,how='left',on=['uuid'])
preliminary_list_train=preliminary_list_train.merge(egemaps_feats,how='left',on=['uuid'])

preliminary_list_test=preliminary_list_test.merge(egemaps_pre,how='left',on=['uuid'])
preliminary_list_test=preliminary_list_test.merge(tsv_feats,how='left',on=['uuid'])
preliminary_list_test=preliminary_list_test.merge(egemaps_feats,how='left',on=['uuid'])

##标签映射：
label_dict={'CTRL':0,'AD':1,'MCI':2}
preliminary_list_train['label']=preliminary_list_train['label'].map(label_dict)
sex_mapping = {'F':0, 'M':1}
preliminary_list_train['sex'] = preliminary_list_train['sex'].map(sex_mapping)
preliminary_list_test['sex'] = preliminary_list_test['sex'].map(sex_mapping)
from numpy import *
# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        print(centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment
# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法

myCentroids,clustAssing = kMeans(preliminary_list_train.drop(['uuid','label'],3))
print(myCentroids)
print(clustAssing)


ed
#===========================分割线=============================================
##模型训练预测
#from sklearn import preprocessing
#lbl = preprocessing.LabelEncoder()
#preliminary_list_train['sex'] = lbl.fit_transform(preliminary_list_train['sex'].astype(int))#将提示的包含错误数据类型这一列进行转换
#preliminary_list_train['sex']=preliminary_list_train['sex'].astype(int)
#preliminary_list_train=preliminary_list_train.drop(columns=['sex'])

prediction_lgb=\
      lgb_model(preliminary_list_train.drop(['uuid','label'],axis=1),\
                preliminary_list_train['label'],\
                preliminary_list_test.drop(['uuid','label'],axis=1))
preliminary_list_test['label']=prediction_lgb
print(preliminary_list_test['label'].value_counts())
label_dict1={0:'CTRL',1:'AD',2:'MCI'}
preliminary_list_test['label']=preliminary_list_test['label'].map(label_dict1)
preliminary_list_test[['uuid','label']].to_csv('submit.csv',index=None)
