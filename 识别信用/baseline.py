#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import catboost as ctb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold,StratifiedKFold
import time
from tqdm import tqdm
import itertools 
import gc
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
def one_hot_col(col):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl
def xgb_model(new_train,y,new_test,lr,N):
    xgb_params = {'booster': 'gbtree',
          'eta':lr, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective':'binary:logistic',
          'eval_metric': 'auc',
          'silent': True,
          }
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=42)
    oof_xgb=np.zeros(new_train.shape[0])
    prediction_xgb=np.zeros(new_test.shape[0])
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = xgb.DMatrix(new_train[tr],y[tr])
        dvalid = xgb.DMatrix(new_train[va],y[va])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
        bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=200, \
        verbose_eval=100, params=xgb_params)
        oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
        prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)
        del bst,dtrain,dvalid
        gc.collect()
    print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))
    prediction_xgb/=N
    return oof_xgb,prediction_xgb
def lgb_model(new_train,y,new_test,N,ff,bf,n):
    params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': ff,
    'bagging_fraction': bf,
    'bagging_freq': 5,
    'num_leaves': n,
    'verbose': -1,
    'max_depth': -1,
  #  'reg_alpha':2.2,
  #  'reg_lambda':1.4,
    'seed':42,
    }
    params1 = { 
        'objective':'reg:logistic',
        'learning_rate': 0.1, 
        #'n_estimators': 4, 
        'max_depth': 5, 
        'min_child_weight': 1, 
        'seed': 0,
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'gamma': 1, 
        'reg_alpha': 1, 
        'reg_lambda': 1,
        'silent': True,
        'eval_metric': 'auc',
    'verbose_eval':50
               }

    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=42)
    oof_lgb=np.zeros(new_train.shape[0]) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros(new_test.shape[0])  ##用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = lgb.Dataset(new_train[tr],y[tr])
        dvalid = lgb.Dataset(new_train[va],y[va],reference=dtrain)
        ##训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid,\
                        verbose_eval=100,early_stopping_rounds=200)
        ##预测验证集：
        oof_lgb[va] += bst.predict(new_train[va], num_iteration=bst.best_iteration)
        ##预测测试集：
        prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)
        
    
    print('the roc_auc_score for train:',roc_auc_score(y,oof_lgb)) ##线下auc评分
    prediction_lgb/=N
    return oof_lgb,prediction_lgb,feature_importance_df
def lgb_model12(new_train,y,new_test,N,ff,bf,n):
    params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': ff,
    'bagging_fraction': bf,
    'bagging_freq': 5,
    'num_leaves': n,
    'verbose': -1,
    'max_depth': -1,
  #  'reg_alpha':2.2,
  #  'reg_lambda':1.4,
    'seed':42,
    }
    params1 = { 
        'objective':'reg:logistic',
        'learning_rate': 0.1, 
        #'n_estimators': 4, 
        'max_depth': 5, 
        'min_child_weight': 1, 
        'seed': 0,
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'gamma': 1, 
        'reg_alpha': 1, 
        'reg_lambda': 1,
        'silent': True,
        'eval_metric': 'auc',
    'verbose_eval':50
               }

    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=42)
    oof_lgb=np.zeros(new_train.shape[0]) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros(new_test.shape[0])  ##用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = lgb.Dataset(new_train.loc[tr],y[tr])
        dvalid = lgb.Dataset(new_train.loc[va],y[va],reference=dtrain)
        ##训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid,\
                        verbose_eval=100,early_stopping_rounds=200)
        ##预测验证集：
        oof_lgb[va] += bst.predict(new_train.loc[va], num_iteration=bst.best_iteration)
        ##预测测试集：
        prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(new_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    
    print('the roc_auc_score for train:',roc_auc_score(y,oof_lgb)) ##线下auc评分
    prediction_lgb/=N
    return oof_lgb,prediction_lgb,feature_importance_df
def lgb_feature1(new_train,y,new_test,lr,N):
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=2019)
    oof_xgb=np.zeros(new_train.shape[0])
    prediction_xgb=np.zeros(new_test.shape[0])
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        model =lgb.LGBMClassifier(boosting_type='gbdt',  min_data_in_leaf=5,\
                                  max_bin=200, num_leaves=16, learning_rate=lr,\
                                  random_state=42,n_estimators=15000,verbose_eval=200)
        model.fit(new_train[tr],y[tr],\
                  eval_set=(new_train[va],y[va]),\
                  early_stopping_rounds=200,\
                  verbose=200)
    
        oof_xgb[va] +=model.predict_proba(new_train[va])[:,1]
        prediction_xgb += model.predict_proba(new_test)[:,1]
    print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))

    prediction_xgb/=N
    return oof_xgb,prediction_xgb
def gbdt_feature1(new_train,y,new_test,N):
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=2019)
    oof_xgb=np.zeros(new_train.shape[0])
    prediction_xgb=np.zeros(new_test.shape[0])
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        model= GradientBoostingClassifier(n_estimators=799,\
                                         learning_rate=0.1)
        model.fit(new_train[tr],y[tr])
    
        oof_xgb[va] +=model.predict(new_train[va])
        prediction_xgb += model.predict(new_test)
    print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))

    prediction_xgb/=N
    return oof_xgb,prediction_xgb
def lgb_feature2(new_train,y,new_test,lr,N,seed):
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=seed)
    oof_xgb=np.zeros(new_train.shape[0])
    prediction_xgb=np.zeros(new_test.shape[0])
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        model =lgb.LGBMClassifier(boosting_type='gbdt',  min_data_in_leaf=5,\
                                  max_bin=200, num_leaves=25, learning_rate=lr,\
                                  random_state=seed,n_estimators=15000)
        model.fit(new_train[tr],y[tr],\
                  eval_set=(new_train[va],y[va]),\
                  early_stopping_rounds=200,\
                  verbose=200)
    
        oof_xgb[va] +=model.predict_proba(new_train[va])[:,1]
        prediction_xgb += model.predict_proba(new_test)[:,1]
    print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))

    prediction_xgb/=N
    return oof_xgb,prediction_xgb
def cat_model(new_train,y,new_test,lr,N,cate_feat_indx):
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=2019)
    oof_xgb=np.zeros(new_train.shape[0])
    prediction_xgb=np.zeros(new_test.shape[0])
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        model = ctb.CatBoostClassifier(iterations=5000,learning_rate=lr,max_depth=5,verbose=100,
                                       early_stopping_rounds=200,\
                                       random_seed=2019,
                                       eval_metric='AUC',
                                       #task_type="GPU"
                                       )
        model.fit(new_train[tr],y[tr],cat_features=cate_feat_indx,\
                  eval_set=(new_train[va],y[va]))
    
        oof_xgb[va] +=model.predict_proba(new_train[va])[:,1]
        prediction_xgb += model.predict_proba(new_test)[:,1]
    print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))

    prediction_xgb/=N
    return oof_xgb,prediction_xgb

train= pd.read_csv('train.csv')
train_label= pd.read_csv('train_label.csv')
test= pd.read_csv('test.csv')
#submission=pd.read_csv('submission.csv')

print(train_label.Label.value_counts())
train=train.merge(train_label,how='left',on=['ID'])
test['Label']=-1

data=pd.concat([train,test],axis=0).reset_index(drop=True)

useful_cols=[]
for col in data.columns:
    if data[col].nunique()==1:
        pass
    else:
        useful_cols.append(col)
data=data[useful_cols]
del useful_cols

data['邮政编码']=data['邮政编码'].astype(str)
data['邮政编码']=data['邮政编码'].apply(lambda x:np.int(x.split('.')[0]) if len(x)>5 else 0)
data['邮政编码_2']=data['邮政编码'].apply(lambda x:np.int(str(x)[:2]) if len(str(x))>5 else 0)
data['邮政编码_4']=data['邮政编码'].apply(lambda x:np.int(str(x)[2:4]) if len(str(x))>5 else 0)

data['经营期限至_isnan']=data['经营期限至'].apply(lambda x:1 if len(str(x))<4 else 0)
data['核准日期_isnan']=data['核准日期'].apply(lambda x:1 if len(str(x))<4 else 0)
data['注销时间_isnan']=data['注销时间'].apply(lambda x:1 if len(str(x))<4 else 0)
data['成立日期_isnan']=data['成立日期'].apply(lambda x:1 if len(str(x))<4 else 0)
data['经营期限自_isnan']=data['经营期限自'].apply(lambda x:1 if len(str(x))<4 else 0)

data['经营期限至_m']=data['经营期限至'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[0]))
data['核准日期_m']=data['核准日期'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[0]))
data['注销时间_m']=data['注销时间'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[0]))
data['成立日期_m']=data['成立日期'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[0]))
data['经营期限自_m']=data['经营期限自'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[0]))
'''
data['经营期限至_s']=data['经营期限至'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[1].split('.')[0]))
data['核准日期_s']=data['核准日期'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[1].split('.')[0]))
data['注销时间_s']=data['注销时间'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[1].split('.')[0]))
data['成立日期_s']=data['成立日期'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[1].split('.')[0]))
data['经营期限自_s']=data['经营期限自'].apply(lambda x:-1 if len(str(x))<4 else int(x.split(':')[1].split('.')[0]))
'''

data['经营范围']=data['经营范围'].apply(lambda x:list(set(map(int,x.split('[')[1].split(']')[0].split(',')))))
data['经营范围_len']=data['经营范围'].apply(lambda x:len(x))

data['ID']=data['ID'].astype(str)
cate_id_dict={}
for i,row in data.iterrows():
    for e in row['经营范围']:
        cate_id_dict.setdefault(e,[]).append(row['ID'])


data['ID']=data['ID'].astype(int)
cols_indx = np.where(data.dtypes != np.object)[0]
cols=[data.columns[i] for i in cols_indx]
data=data[cols]
train=data[:len(train)]
test=data[len(train):].reset_index(drop=True)
#del data
oof_lgb,prediction=\
      xgb_model(np.array(train.drop(['ID','Label'],axis=1)),\
                train.Label.values,\
                np.array(test.drop(['ID','Label'],axis=1)),\
                0.4,5)
ed
oof_lgb1,prediction1,inp=\
      lgb_model(np.array(train.drop(['ID','Label'],axis=1)),\
                train.Label.values,\
                np.array(test.drop(['ID','Label'],axis=1)),\
                5,0.8,0.8,16)
test['Label']=prediction
test[['ID','Label']].to_csv('sub.csv', index=False)