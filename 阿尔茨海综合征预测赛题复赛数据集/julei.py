# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:46:00 2019

@author: Sun
"""

import numpy as np
import math
import copy
from collections import defaultdict
from sklearn.cluster import KMeans


class GEM:
    def __init__(self, maxstep=1000, epsilon=1e-3, K=3):
        self.maxstep = maxstep
        self.epsilon = epsilon
        self.K = K  # 混合模型中的分模型的个数

        self.alpha = None  # 每个分模型前系数
        self.mu = None  # 每个分模型的均值向量
        self.sigma = None  # 每个分模型的协方差
        self.gamma_all_final = None  # 存储最终的每个样本对分模型的响应度，用于最终的聚类

        self.D = None  # 输入数据的维度
        self.N = None  # 输入数据总量

    def inin_param(self, data):
        # 初始化参数
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.init_param_helper(data)
        return

    def init_param_helper(self, data):
        # KMeans初始化模型参数
        KMEANS = KMeans(n_clusters=self.K).fit(data)
        clusters = defaultdict(list)
        for ind, label in enumerate(KMEANS.labels_):
            clusters[label].append(ind)
        mu = []
        alpha = []
        sigma = []
        for inds in clusters.values():
            partial_data = data[inds]
            mu.append(partial_data.mean(axis=0))  # 分模型的均值向量
            alpha.append(len(inds) / self.N)  # 权重
            sigma.append(np.cov(partial_data.T))  # 协方差,D个维度间的协方差
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        return

    def _phi(self, y, mu, sigma):
        # 获取分模型的概率
        
        s1 = 1.0 / (math.sqrt(np.linalg.det(sigma))+1)
        s2 = np.linalg.inv(sigma+1)  # d*d
        delta = np.array([y - mu])  # 1*d
        return s1 * math.exp(-1.0 / 2 * delta @ s2 @ delta.T)

    def fit(self, data):
        # 迭代训练
        self.inin_param(data)
        step = 0
        gamma_all_arr = None
        while step < self.maxstep:
            step += 1
            old_alpha = copy.copy(self.alpha)
            # E步
            gamma_all = []
            for j in range(self.N):
                gamma_j = []    # 依次求每个样本对K个分模型的响应度

                for k in range(self.K):
                    gamma_j.append(self.alpha[k] * self._phi(data[j], self.mu[k], self.sigma[k]))

                s = sum(gamma_j)
                gamma_j = [item/s for item in gamma_j]
                gamma_all.append(gamma_j)

            gamma_all_arr = np.array(gamma_all)
            # M步
            for k in range(self.K):
                gamma_k = gamma_all_arr[:, k]
                SUM = np.sum(gamma_k)
                # 更新权重
                self.alpha[k] = SUM / self.N  # 更新权重
                # 更新均值向量
                new_mu = sum([gamma * y for gamma, y in zip(gamma_k, data)]) / SUM  # 1*d
                self.mu[k] = new_mu
                # 更新协方差阵
                delta_ = data - new_mu   # n*d
                self.sigma[k] = sum([gamma * (np.outer(np.transpose([delta]), delta)) for gamma, delta in zip(gamma_k, delta_)]) / SUM  # d*d
            alpha_delta = self.alpha - old_alpha
            if np.linalg.norm(alpha_delta, 1) < self.epsilon:
                break
        self.gamma_all_final = gamma_all_arr
        return

    def predict(self,list):
        cluster = defaultdict(list)
        for j in range(self.N):
            max_ind = np.argmax(self.gamma_all_final[j])
            cluster[max_ind].append(j)
        return cluster

if __name__ == '__main__':
    '''
    def generate_data(N=500):
        X = np.zeros((N, 2))  # N*2, 初始化X
        mu = np.array([[5, 35], [20, 40], [20, 35], [45, 15]])
        sigma = np.array([[30, 0], [0, 25]])
        for i in range(N): # alpha_list=[0.3, 0.2, 0.3, 0.2]
            prob = np.random.random(1)
            if prob < 0.1:  # 生成0-1之间随机数
                X[i, :] = np.random.multivariate_normal(mu[0], sigma, 1)  # 用第一个高斯模型生成2维数据
            elif 0.1 <= prob < 0.3:
                X[i, :] = np.random.multivariate_normal(mu[1], sigma, 1)  # 用第二个高斯模型生成2维数据
            elif 0.3 <= prob < 0.6:
                X[i, :] = np.random.multivariate_normal(mu[2], sigma, 1)  # 用第三个高斯模型生成2维数据
            else:
                X[i, :] = np.random.multivariate_normal(mu[3], sigma, 1)  # 用第四个高斯模型生成2维数据
        return X
    '''
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
    import jieba
    import re
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
        z['duration'] = z.end_time - z.start_time
        tsv1 = analysis_df(z)
        df = pd.DataFrame([tsv1])
        tsv2=text_segmentation_one_tsv(z)
        df1 = pd.DataFrame([tsv2])
   
    
    ##说一句话所用时长：
        z['end_time-start_time']=z['end_time']-z['start_time']
        z['jicha'] = z['end_time-start_time'].max()-z['end_time-start_time'].min()
    
        tsv_feats.append([path[:-4],\
                          z['end_time-start_time'].mean(),\
                          z['end_time-start_time'].min(),\
                          z['end_time-start_time'].max(),\
                          z['end_time-start_time'].std(),\
                          z['end_time-start_time'].skew(),\
                          z['end_time-start_time'].var(),\
                          z['jicha'].mean(),\
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
    label_dict={'CTRL':0,'AD':2,'MCI':1}
    preliminary_list_train['label']=preliminary_list_train['label'].map(label_dict)
    sex_mapping = {'F':0, 'M':1}
    preliminary_list_train['sex'] = preliminary_list_train['sex'].map(sex_mapping)
    preliminary_list_test['sex'] = preliminary_list_test['sex'].map(sex_mapping)
    print('特征选择...')
    print(preliminary_list_train.shape)
    train_y=preliminary_list_train['label']
    train_x=preliminary_list_train.drop(['uuid','label'],axis=1)
    data=np.array(train_x)

    gem = GEM()
    gem.fit(data)
    # print(gem.alpha, '\n', gem.sigma, '\n', gem.mu)
    cluster = gem.predict(data)

    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle('grbk')
    for color, inds in zip(colors, cluster.values()):
        partial_data = data[inds]
        plt.scatter(partial_data[:,0], partial_data[:, 1], edgecolors=color)
    plt.show()
