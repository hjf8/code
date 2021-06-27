# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:48:29 2019

@author: shaowu
"""

import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm,tqdm_notebook 
## 字符串处理工具包
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from functools import partial

import os 
import gc
def huafen_data(tr):
    '''划分数据集：（val_index,tr_index），按顺序划分'''
    return [([i for i in range(0,len(tr),5)],[i for i in range(len(tr))\
                                                      if i not in [i for i in range(0,len(tr),5)]]),\
       ([i for i in range(1,len(tr),5)],[i for i in range(len(tr))\
                                                      if i not in [i for i in range(1,len(tr),5)]]),\
       ([i for i in range(2,len(tr),5)],[i for i in range(len(tr))\
                                                      if i not in [i for i in range(2,len(tr),5)]]),\
       ([i for i in range(3,len(tr),5)],[i for i in range(len(tr))\
                                                      if i not in [i for i in range(3,len(tr),5)]]),\
       ([i for i in range(4,len(tr),5)],[i for i in range(len(tr))\
                                                      if i not in [i for i in range(4,len(tr),5)]])
       ]

def huafen_dataset(tr,k,step):
    '''划分数据集tr：（val_index,tr_index），按顺序划分k份'''
    pers=[]
    while k>0:
        pers.append(([i for i in range(step-k,len(tr),step)],[i for i in range(len(tr))\
                                                      if i not in [i for i in range(step-k,len(tr),step)]]))
        k=k-1
    return pers

train = pd.read_csv('train_dataset.csv')
test = pd.read_csv('test_dataset.csv')

train=train[(train['信用分']>430)&(train['信用分']<711)]
#根据信用分，从小到大排序train
train=train.sort_index(by='信用分').reset_index(drop=True)

def _simple_features(df_):
    df = df_.copy() 
    df['次数'] = df['当月网购类应用使用次数'] +  df['当月物流快递类应用使用次数'] +  df['当月金融理财类应用使用总次数'] + df['当月视频播放类应用使用次数']\
                 + df['当月飞机类应用使用次数'] + df['当月火车类应用使用次数'] + df['当月旅游资讯类应用使用次数']  + 1
        
    for col in ['当月金融理财类应用使用总次数','当月旅游资讯类应用使用次数']: # 这两个比较积极向上一点
        df[col + '百分比'] = df[col].values / df['次数'].values 
    
    
    df['当月通话人均话费'] = df['用户账单当月总费用（元）'].values / (df['当月通话交往圈人数'].values + 1)
    df['上个月费用'] = df['用户当月账户余额（元）'].values + df['用户账单当月总费用（元）'].values
    
    #df['f4'] = np.log(df['用户网龄（月）'])
    
    df['用户上网年龄'] = df['用户年龄'] - df['用户网龄（月）']
    df['用户上网年龄百分比'] = df['用户网龄（月）'] / (df['用户年龄'] + 1)
     
    df['近似总消费'] = df['用户近6个月平均消费值（元）'] / 6 * df['用户网龄（月）']
    
    df['f2'] = (df['缴费用户最近一次缴费金额（元）'])*(df['用户最近一次缴费距今时长（月）']+1)
    #df['f3'] = df['用户账单当月总费用（元）']+(df['用户当月账户余额（元）'])-df['缴费用户最近一次缴费金额（元）']
    df['f3'] = df['用户账单当月总费用（元）']-(df['用户近6个月平均消费值（元）'])
    
    
    #df['f3']=df['f3'].apply(lambda x:1 if x>=0 else 0)
    #df['f1'] = np.log((df['用户近6个月平均消费值（元）'])* (df['当月通话交往圈人数']+1))
    #df['f2'] = df['用户账单当月总费用（元）']* (df['当月通话交往圈人数']+1)
    #df['f1'] = np.log(df['用户近6个月平均消费值（元）'])* np.log(df['当月通话交往圈人数'])
    #df['f2'] = np.log(df['用户账单当月总费用（元）'])* np.log(df['当月通话交往圈人数'])
    #df['f3'] = np.log(df['当月通话交往圈人数'])
    feats2=['用户年龄',
       '用户网龄（月）', '用户最近一次缴费距今时长（月）', '缴费用户最近一次缴费金额（元）', 
       '用户近6个月平均消费值（元）',
       '用户账单当月总费用（元）', '用户当月账户余额（元）', 
       '当月通话交往圈人数',
       '近三个月月均商场出现次数','当月网购类应用使用次数', 
       '当月物流快递类应用使用次数',
       '当月金融理财类应用使用总次数', '当月视频播放类应用使用次数', 
       '当月飞机类应用使用次数', '当月火车类应用使用次数',
       '当月旅游资讯类应用使用次数',]
    from itertools import combinations
    cols_2=list(combinations(feats2, 2))
    '''
    for i in cols_2:
        #df[i[0]+'+'+i[1]]=np.log((df[i[0]]+df[i[1]]))
        #df[i[0]+'-'+i[1]]=np.log((df[i[0]]-df[i[1]]))
        df[i[0]+'*'+i[1]]=np.log((df[i[0]]*df[i[1]]))
        df[i[0]+'/'+i[1]]=np.log((df[i[0]]/(df[i[1]]+1)))
    '''
    return df
    
train_fea = _simple_features(train)
test_fea  = _simple_features(test)
'''
relation=train_fea.corr()
length = relation.shape[0]
high_corr = list()
final_cols = []
del_cols =[]
for i in range(length):
    if relation.columns[i] not in del_cols:
        final_cols.append(relation.columns[i])
        for j in range(i+1,length):
            if (relation.iloc[i,j] > 0.95) and (relation.columns[j] not in del_cols):
                del_cols.append(relation.columns[j])      
final_cols.append('用户编码')
train_fea = train_fea[final_cols]
#final_cols.remove('label')
final_cols.remove('信用分')
test_fea=test_fea[final_cols]
'''
fea_cols = [col for col in train_fea.columns if train_fea[col].dtypes!='object' and train_fea[col].dtypes != '<M8[ns]' and col!='用户编码' and col!='信用分']
len(fea_cols)
#fea_cols.remove('是否大学生客户')
#fea_cols.remove('用户实名制是否通过核实')
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
def _get_values_lgbregresser_models(df_fea, df_label,  feature_names):
    #kf = KFold(n_splits=5,shuffle=False,random_state=2019)
    #pers=huafen_data(df_fea)
    pers=huafen_dataset(df_fea,10,10)
    models  = [] 
    models_1 = []
    models_2 = []
    
    importances = pd.DataFrame() 
    
    lgb_params = {'num_leaves': 31,
         'min_data_in_leaf': 32,#32, 
#          'objective':'mae',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         'n_estimators': 20000,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 50,
         "verbosity": -1}


    lgb_params1 = {'num_leaves': 31,
         'min_data_in_leaf': 32,#32, 
         'objective':'mae',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         'n_estimators': 20000,
         "bagging_seed": 11,
         "lambda_l1": 0.1,
         "nthread": 50,
         "verbosity": -1}
    xgb_params1 = {'learning_rate': 0.05, 'n_estimators': 10000, 'max_depth': 5,\
                'min_child_weight': 32, 'seed':2019,"nthread": 50}
    xgb_params = {'learning_rate': 0.005, 'n_estimators': 10000, 'max_depth': 5,\
                'min_child_weight': 32, 'seed':2019,"nthread": 50}
    min_val = np.min(df_label)
    print(min_val)
  #  for fold_, (trn_, val_) in enumerate(kf.split(df_fea)): 
    for per in pers:
    #    trn_x, trn_y= df_fea[trn_,:], df_label[trn_]#, df_label1[trn_] 
    #    val_x, val_y = df_fea[val_,:], df_label[val_]#, df_label1[val_] 
        
        trn_x, trn_y= df_fea[per[1]], df_label[per[1]]#, df_label1[trn_] 
        val_x, val_y = df_fea[per[0]], df_label[per[0]]#, df_label1[val_] 
        tmp = pd.DataFrame()
         
        
        model = lgb.LGBMRegressor(**lgb_params1)
        #model =xgb.XGBRegressor(**xgb_params1)
        model.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric ='mae',
                                          verbose=1000,early_stopping_rounds=250)     
        tmp['target'] = val_y
        tmp['pred1'] = model.predict(val_x)
        models.append(model)
        
        model1 = lgb.LGBMRegressor(**lgb_params)
        #model1 =xgb.XGBRegressor(**xgb_params)
        model1.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)], eval_metric ='mae',
                                           verbose=1000,early_stopping_rounds=250)     
        tmp['pred2'] = model1.predict(val_x)
        models_1.append(model1)
  
        tmp = tmp.sort_values('pred1')
        tmp['ranks'] = list(range(tmp.shape[0]))
        tmp['preds'] = tmp['pred1'].values
        tmp.loc[tmp.ranks<2000,'preds']  = tmp.loc[tmp.ranks< 2000,'pred2'].values *0.4 + tmp.loc[tmp.ranks< 2000,'pred1'].values * 0.6
        tmp.loc[tmp.ranks>8000,'preds']  = tmp.loc[tmp.ranks> 8000,'pred2'].values *0.4 + tmp.loc[tmp.ranks> 8000,'pred1'].values * 0.6
         
        print('*' * 100)
        print('MAE Model',     1 / (1 + (mean_absolute_error(y_true= tmp['target'] , y_pred= tmp['pred1'] ))))
        print('MSE Model',     1 / (1 + (mean_absolute_error(y_true= tmp['target'] , y_pred= tmp['pred2'] ))))
        print('Merge Model12', 1 / (1 + (mean_absolute_error(y_true= tmp['target'] , y_pred= tmp['preds'] )))) 
        
        imp_df = pd.DataFrame()
        imp_df['feature'] = feature_names
        #print(model.get_fscore())
        
        imp_df['gain'] = model.feature_importances_
        
        #imp_df['fold'] = fold_ + 1
        
        importances = pd.concat([importances, imp_df], axis=0)
        
        gc.collect() 
    return models,models_1,importances

#models_mae, models_mse, importances= _get_values_lgbregresser_models(train_fea[fea_cols].values, train_fea['信用分'].values, feature_names=fea_cols)
models_mae, models_mse,importances= _get_values_lgbregresser_models(train_fea[fea_cols].values, train_fea['信用分'].values, feature_names=fea_cols)


pred_mae = 0
for i,model in enumerate(models_mae): 
    #if i not in [5,8,9]:
    pred_mae += model.predict(test_fea[fea_cols].values)/10
    #pred_mae += model.predict(test_fea[fea_cols]) * 0.2
    #pred_mae += model.predict(test_fea[fea_cols].values) * 0.1
test_fea['pred_mae'] = pred_mae
pred_mse = 0
for i,model in enumerate(models_mse): 
  #  if i not in [5,8,9]:
    pred_mse += model.predict(test_fea[fea_cols].values)/10
    #pred_mse += model.predict(test_fea[fea_cols]) * 0.2
    #pred_mse += model.predict(test_fea[fea_cols].values) * 0.1
test_fea['pred_mse'] = pred_mse

test_fea = test_fea.sort_values('pred_mae')
test_fea['ranks'] = list(range(test_fea.shape[0]))
test_fea['score'] = test_fea['pred_mae'].values
test_fea.loc[test_fea.ranks<10000,'score']  = test_fea.loc[test_fea.ranks< 10000,'pred_mse'].values *0.4 + test_fea.loc[test_fea.ranks< 10000,'pred_mae'].values * 0.6
test_fea.loc[test_fea.ranks>40000,'score']  = test_fea.loc[test_fea.ranks> 40000,'pred_mse'].values *0.4 + test_fea.loc[test_fea.ranks> 40000,'pred_mae'].values * 0.6
submit_mae_mse = pd.DataFrame()
submit_mae_mse['id']    = test_fea['用户编码'].values
submit_mae_mse['score'] = test_fea['score'].values 
submit_mae_mse['score'] = submit_mae_mse['score'].astype(int)
submit_mae_mse[['id','score']].to_csv('baseline_mae_mse.csv',index = None)
submit_mae_mse['score'].describe()
