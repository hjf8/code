# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:31:46 2019

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
import jieba
import re
def lgb_model11(new_train,y,new_test):
    params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_leaves': 1000,
    'verbose': -1,
    'max_depth': -1,
  #  'reg_alpha':2.2,
  #  'reg_lambda':1.4,
    'seed':42,
    }
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_lgb=np.zeros(new_train.shape[0]) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros(new_test.shape[0])  ##用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = lgb.Dataset(new_train.loc[tr],y[tr])
        dvalid = lgb.Dataset(new_train.loc[va],y[va],reference=dtrain)
        ##训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid, verbose_eval=400,early_stopping_rounds=200)
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
    print('the recall_score for train:',recall_score(y,[1 if i>0.5 else 0 for i in oof_lgb], average='macro'))
    print('the f1_score for train:',f1_score(y,[1 if i>0.5 else 0 for i in oof_lgb],average='weighted'))

    prediction_lgb/=5
    return oof_lgb,prediction_lgb,feature_importance_df   

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
    #tsv1 = analysis_df(z)
    #df = pd.DataFrame([tsv1])
    #tsv2=text_segmentation_one_tsv(z)
    #df1 = pd.DataFrame([tsv2])
   
    
    ##说一句话所用时长：
    z['end_time-start_time']=z['end_time']-z['start_time']
    z['jicha'] = z['end_time-start_time'].max()-z['end_time-start_time'].min()
    
    tsv_feats.append([path[:-4],\
                      z['end_time-start_time'].mean(),\
                      z['end_time-start_time'].min(),\
                      z['end_time-start_time'].max(),\
                      z['end_time-start_time'].std(),\
                      z['end_time-start_time'].median(),\
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
label_dict2={'CTRL':0,'AD':2,'MCI':1}
preliminary_list_train['label']=preliminary_list_train['label'].map(label_dict2)
sex_mapping2 = {'F':0, 'M':1}
preliminary_list_train['sex'] = preliminary_list_train['sex'].map(sex_mapping2)
preliminary_list_test['sex'] = preliminary_list_test['sex'].map(sex_mapping2)
#print('特征选择...')
#print(preliminary_list_train.shape)




train1= pd.read_csv("data_final/1_preliminary_list_train.csv")
test1= pd.read_csv("data_final/1_preliminary_list_test.csv")
#第一部分
train1.drop(['label'],inplace=True,axis=1)
test1.drop(['label'],inplace=True,axis=1)
train1=train1.merge(preliminary_list_train,how='left',on=['uuid'])
test1=test1.merge(preliminary_list_test,how='left',on=['uuid'])


##标签映射：
#label_dict={'CTRL':0,'AD':1}
#train1['label']=train1['label'].map(label_dict)
#sex_mapping = {'F':0, 'M':1}
#train1['sex'] = train1['sex'].map(sex_mapping)
#test1['sex'] = test1['sex'].map(sex_mapping)

#print('特征选择...')
#print(train1.shape)



train1.fillna(0,inplace=True)
oof_lgb,prediction_lgb,feature_importance_df=\
      lgb_model11(train1.drop(['uuid','label'],axis=1),\
                train1['label'],\
                test1.drop(['uuid','label'],axis=1))
result=[1 if i>0.5 else 0 for i in prediction_lgb]
test1['label']=result
print(test1['label'].value_counts())
test1['label']=test1['label'].apply(lambda x:'AD' if x==1 else 'CTRL')      

#第二部分
#train2=preliminary_list_train[~preliminary_list_train.uuid.isin(train1.uuid.unique())].reset_index(drop=True)
test2=preliminary_list_test[~preliminary_list_test.uuid.isin(test1.uuid.unique())].reset_index(drop=True)
#label_dict1={'CTRL':0,'AD':2,'MCI':1}
#train2['label']=train2['label'].map(label_dict1)
#sex_mapping1 = {'F':0, 'M':1}
#train2['sex'] = train2['sex'].map(sex_mapping1)
#test2['sex'] = test2['sex'].map(sex_mapping)
#train3=train2[:72]
#train4 = pd.concat([train1,train3]).reset_index(drop=True)
#preliminary_list_train=train4[['uuid']].merge(preliminary_list_train,how='left',on=['uuid'])
#preliminary_list_train['label']=preliminary_list_train['label'].map(label_dict1)

#preliminary_list_train['sex'] = preliminary_list_train['sex'].map(sex_mapping1)
#preliminary_list_test['sex'] = preliminary_list_test['sex'].map(sex_mapping1)
print('特征选择...')



'''
from sklearn.model_selection import train_test_split
train_y=preliminary_list_train['label']
train_x=preliminary_list_train.drop(['uuid','label'],axis=1)
x, val, y, val_y= train_test_split(train_x,train_y,test_size=0.4,random_state=2018,stratify=train_y)
test=preliminary_list_test.drop(['uuid','label'],axis=1)
x=x.values#转换成array数据类型
val=val.values
test=test.values
'''
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors,tree
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier

def gbc_model(new_train,y,new_test):

    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=2019)
    oof_lgb=np.zeros((new_train.shape[0],3)) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros((new_test.shape[0],3))  ##用于存放测试集概率，k折最后要除以k取平均
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.8,
                                     max_depth=6,random_state=2018)
        model.fit(new_train.loc[tr],y[tr])
        ##预测验证集：
        oof_lgb[va] =model.predict_proba(new_train.loc[va])
        ##预测测试集：
        #prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)
        prediction_lgb += model.predict_proba(new_test)
    return np.argmax(prediction_lgb, axis=1),prediction_lgb
'''
preliminary_list_train.fillna(0,inplace=True)
prediction_lgb1,prediction_lgb11= gbc_model(preliminary_list_train.drop(['uuid','label'],axis=1),\
                preliminary_list_train['label'],\
                test2.drop(['uuid','label'],axis=1))
'''
'''
#from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,RandomForestRegressor
#from sklearn.feature_selection import SelectFromModel
#cols=[i for i in preliminary_list_train.columns if i not in ['uuid','label']]
#sm = SelectFromModel(GradientBoostingClassifier(random_state=2019))
#data_new= sm.fit_transform(preliminary_list_train.drop(['uuid','label'],axis=1).fillna(-99), preliminary_list_train['label'].values)
#data_new1= sm.transform(preliminary_list_test.drop(['uuid','label'],axis=1).fillna(-99))
#print(data_new.shape)
#print([cols[i] for i in sm.get_support([0])])

prediction_lgb=\
      lgb_model(preliminary_list_train[[cols[i] for i in sm.get_support([0])]],\
                preliminary_list_train['label'],\
                preliminary_list_test[[cols[i] for i in sm.get_support([0])]])
ed
'''
#===========================分割线=============================================
##模型训练预测
#from sklearn import preprocessing
#lbl = preprocessing.LabelEncoder()
#preliminary_list_train['sex'] = lbl.fit_transform(preliminary_list_train['sex'].astype(int))#将提示的包含错误数据类型这一列进行转换
#preliminary_list_train['sex']=preliminary_list_train['sex'].astype(int)
#preliminary_list_train=preliminary_list_train.drop(columns=['sex'])
'''
prediction_lgb,lgb11=\
      lgb_model(preliminary_list_train.drop(['uuid','label'],axis=1),\
                preliminary_list_train['label'],\
                preliminary_list_test.drop(['uuid','label'],axis=1))


preliminary_list_test['label']=prediction_lgb
print(preliminary_list_test['label'].value_counts())
label_dict1={0:'CTRL',2:'AD',1:'MCI'}
preliminary_list_test['label']=preliminary_list_test['label'].map(label_dict1)
preliminary_list_test[['uuid','label']].to_csv('submit.csv',index=None)
ed
'''
preliminary_list_train.fillna(0,inplace=True)
prediction_lgb1,lgb11=\
      gbc_model(preliminary_list_train.drop(['uuid','label'],axis=1),\
                preliminary_list_train['label'],\
                test2.drop(['uuid','label'],axis=1))
  
    
test2['label']=prediction_lgb1

label_dict2={0:'CTRL',2:'AD',1:'MCI'}
test2['label']=test2['label'].map(label_dict2)
test3=pd.concat([test1,test2]).reset_index(drop=True)
preliminary_list_test=preliminary_list_test[['uuid']].merge(test3,how='left',on=['uuid'])
print(preliminary_list_test['label'].value_counts())
#preliminary_list_test[['uuid','label']].to_csv('submit.csv',index=None)
aa=pd.read_csv("78.csv")
for i in range(len(aa)):
    if aa.loc[i,'label']!=preliminary_list_test.loc[i,'label']:
        print(preliminary_list_test.loc[i,'uuid'])