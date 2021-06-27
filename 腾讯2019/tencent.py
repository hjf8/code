# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:19:20 2019

@author: shaowu
"""
import gc
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal as signal
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm import *
#files=os.listdir('mimic/')

##--------------------xgboost模型 5cv----------------------------------------
def xgb_model(X_train,y_train):
    xgb_params = {'eta': 0.01, 'max_depth': 5, 'subsample': 1, 'colsample_bytree': 1, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}
    
    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(X_train))
    #predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist,\
                        early_stopping_rounds=200, verbose_eval=500,params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        #predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    
    print("CV score: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_xgb, y_train))))
    return oof_xgb

##--------------------lightgbm模型 5cv----------------------------------------
def lgb_model(X_train,y_train):
    #test=test[X_train.columns]
    param = {'num_leaves': 128,#120,
         'min_data_in_leaf': 5,#32, 
         'objective':'regression',
         #'objective': 'binary', 
         'max_depth': -1,
         'learning_rate': 0.005,#0.01,
         #"min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 1,
         "bagging_freq": 5,
         "bagging_fraction": 1,
         "bagging_seed": 11,
         #"metric": 'auc',
         "metric": 'mae',#'rmse',
         #"lambda_l1": 0.1,
         "verbosity": -1}
    folds = KFold(n_splits=10, shuffle=True, random_state=2019)
    oof_lgb = np.zeros(len(X_train))
    #predictions_lgb = np.zeros(len(test))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 200)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    
        #predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
    #print("CV score: {:<8.8f}".format(1/(1+mean_squared_error(oof_lgb, y_train))))
    print("CV rmse: {:<8.8f}".format(np.sqrt(mean_squared_error(oof_lgb, y_train))))
    print("CV mae: {:<8.8f}".format(mean_absolute_error(oof_lgb, y_train)))
    return oof_lgb#,importances


    
ad_operation=pd.read_table('ad_operation.dat',header=None)
ad_operation.columns=['广告id','修改时间','操作类型','修改字段','操作后的字段值']

ad_static_feature=pd.read_table('ad_static_feature.out',header=None)
ad_static_feature.columns=['广告id','创建时间','广告账户id','商品id','商品类型',\
                           '广告行业id','素材尺寸']
test_sample=pd.read_table('test_sample.dat',header=None)
test_sample.columns=['样本id','广告id','创建时间','素材尺寸','广告行业id',\
                     '商品类型','商品id','广告账户id','投放时段',\
                           '人群定向','出价(单位分)']
user_data=pd.read_table('user_data',header=None)
user_data.columns=['用户id','年龄','性别','地域','婚恋状态',\
                     '学历','消费能力','设备','工作状态',\
                           '连接类型','行为兴趣']
totalExposureLog=pd.read_table('totalExposureLog.out',header=None)
totalExposureLog.columns=['广告请求id','广告请求时间','广告位id','用户id','曝光广告id',\
                           '曝光广告素材尺寸','曝光广告出价bid','曝光广告pctr',\
                           '曝光广告quality_ecpm','曝光广告totalEcpm']

