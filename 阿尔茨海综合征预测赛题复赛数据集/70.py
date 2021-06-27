# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:11:43 2019

@author: Sun
"""

import os
import pandas as pd
import numpy as np
import time
from tqdm import *
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,recall_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from collections import Counter
import jieba
import re
#三分类模型
def lgb_model(new_train,y,new_test):
    params = {'num_leaves':60,
          'min_data_in_leaf':30,
          'objective':'multiclass',
          'num_class':3,
          'max_depth': -1,
          'learning_rate':0.05,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.8,
          "bagging_freq": 5,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 15,
          'metric': 'multi_logloss',
          "random_state": 2019,
          # 'device': 'gpu'
          }
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=2019)
    oof_lgb=np.zeros((new_train.shape[0],3)) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros((new_test.shape[0],3))  ##用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = lgb.Dataset(new_train.loc[tr],y[tr])
        dvalid = lgb.Dataset(new_train.loc[va],y[va],reference=dtrain)
        ##训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=[dtrain,dvalid], verbose_eval=400,early_stopping_rounds=100)
        ##预测验证集：
        oof_lgb[va] = bst.predict(new_train.loc[va], num_iteration=bst.best_iteration)
        ##预测测试集：
        #prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(new_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration) / skf.n_splits
    return np.argmax(prediction_lgb, axis=1),prediction_lgb
#二分类模型
def xgb_model(new_train,y,new_test,lr,N):
  '''定义模型'''
  xgb_params = {'booster': 'gbtree',
          'eta':lr, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective':'binary:logistic',
          'eval_metric': 'auc',
          'silent': True,
          }
  skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=42)
  oof_xgb=np.zeros(new_train.shape[0])
  prediction_xgb=np.zeros(new_test.shape[0])
  for i,(tr,va) in enumerate(skf.split(new_train,y)):
    print('fold:',i+1,'training')
    dtrain = xgb.DMatrix(new_train[tr],y[tr])
    dvalid = xgb.DMatrix(new_train[va],y[va])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
    bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=200, \
    verbose_eval=400, params=xgb_params)
    oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
    prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)
  print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))
  prediction_xgb/=N
  return oof_xgb,prediction_xgb 
#提取文本特征，结合初赛提供的baseline
def analysis_df(part_df):
    ''' 分析一个DataFrame得到统计量，以字典的形式返回
    '''
    total_rows = len(part_df)
    part_df = part_df.reset_index(drop=True)

    interview_start_time = part_df.loc[0, 'start_time']
    interview_end_time = part_df.loc[total_rows - 1, 'end_time']

    total_duration = interview_end_time-interview_start_time

    A_speak_num = (part_df.speaker == '<A>').sum()
    A_speak_duration_sum = part_df.duration[part_df.speaker == '<A>'].sum()
    A_speak_duration_mean = part_df.duration[part_df.speaker == '<A>'].mean()
    A_speak_duration_std = part_df.duration[part_df.speaker == '<A>'].std()
    A_speak_duration_median = part_df.duration[part_df.speaker == '<A>'].median()
    A_speak_duration_skew = part_df.duration[part_df.speaker == '<A>'].skew()

    B_speak_num = (part_df.speaker == '<B>').sum()
    B_speak_duration_sum  = part_df.duration[part_df.speaker == '<B>'].sum()
    B_speak_duration_mean = part_df.duration[part_df.speaker == '<B>'].mean()
    B_speak_duration_std  = part_df.duration[part_df.speaker == '<B>'].std()
    B_speak_duration_median = part_df.duration[part_df.speaker == '<B>'].median()
    B_speak_duration_skew = part_df.duration[part_df.speaker == '<B>'].skew()

    silence_duration_sum = total_duration - A_speak_duration_sum - B_speak_duration_sum

    result = {
        'total_duration'       : total_duration,
        'A_speak_num'          : A_speak_num,

        'A_speak_duration_mean': A_speak_duration_mean,
        'A_speak_duration_std' : A_speak_duration_std,
        'A_speak_duration_median' : A_speak_duration_median,
        'A_speak_duration_skew' : A_speak_duration_skew,
        
        'B_speak_num'          : B_speak_num,

        'B_speak_duration_mean': B_speak_duration_mean,
        'B_speak_duration_std' : B_speak_duration_std,
        'B_speak_duration_median' : B_speak_duration_median,
        'B_speak_duration_skew' : B_speak_duration_skew,
        # 语音总长占比 语音总长度的比值 语音平均长度的比值 语音方差的比值
        'A_speak_duration_proportion': A_speak_duration_sum / total_duration,
        'B_speak_duration_proportion': B_speak_duration_sum / total_duration,
        'slience_duration_proportion': silence_duration_sum / total_duration,
    }
    return result
# 定义用于识别人工标注的正则表达式
PATTERN_1 = re.compile('【.*?】')  # 方括号注释
PATTERN_2 = re.compile('&')  # 语气词
PATTERN_3 = re.compile('(｛.*?｝)|(\{.*?\})')  # 语法错误
PATTERN_4 = re.compile('\(|\)|（|）')  # 重复修正
PATTERN_5 = re.compile('/')  # 重复修正
PATTERN_6 = re.compile('\?|？|，|。')  # 标点符号
def get_origin_text(annotated_text):
    ''' 对有标注的文本进行处理，得到原始文本
    参数:
        annotated_text : 有标注的文本
    返回:
        origin_text : 删去人工注释的文本 
        num_filledpause : 语气词（有声停顿）的个数  
        num_repeat : 重复的次数
        num_correction : 修正的次数
        num_error : 语法错误的次数
    '''
    origin_text = PATTERN_1.sub('', annotated_text)
    origin_text, num_filledpause       = PATTERN_2.subn('', origin_text)
    origin_text, num_error             = PATTERN_3.subn('', origin_text)
    origin_text, num_correction_repeat = PATTERN_4.subn('', origin_text)
    origin_text, num_slash             = PATTERN_5.subn('', origin_text)
    origin_text = PATTERN_6.sub(' ', origin_text)
    # [\u4e00-\u9fa5]
    num_correction_repeat //= 2
    num_correction = num_slash - num_correction_repeat
    num_repeat = num_correction_repeat - num_correction
    return origin_text, num_filledpause, num_correction, num_repeat, num_error
def text_segmentation_one_tsv(tsv_df):
    ''' 对一个人的文本进行分词
    参数:
        tsv_path:  TSV文件所在目录
        outfile_path: 输出文件目录
    返回:
        sum_filledpause : 语气词（有声停顿）的个数之和
        sum_correction : 修正的次数之和
        sum_repeat : 重复的次数之和
        sum_error : 语法错误的次数之和
    '''
    total_rows = len(tsv_df)
    sum_A_filledpause = 0
    sum_A_correction = 0
    sum_A_repeat = 0
    sum_A_error = 0
    sum_A_word = 0
    sum_B_filledpause = 0
    sum_B_correction = 0
    sum_B_repeat = 0
    sum_B_error = 0
    sum_B_word = 0
    for indx in range(total_rows):
        speaker, value = tsv_df.loc[indx, ['speaker', 'value']]
        if type(speaker) is not str:
            print('warning %s:%d line: speaker:%s' % (tsv_path, indx, str(speaker)))
            continue
        ori_text, num_filledpause, num_correction, num_repeat, num_error = get_origin_text(value)
        if ori_text == '':
            continue
        seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
        seg_list = list(seg_list)
        if speaker.strip() == '<A>':
            sum_A_filledpause += num_filledpause
            sum_A_correction += num_correction
            sum_A_repeat += num_repeat
            sum_A_error += num_error
            sum_A_word += len(seg_list)
        elif speaker.strip() == '<B>':
            sum_B_filledpause += num_filledpause
            sum_B_correction += num_correction
            sum_B_repeat += num_repeat
            sum_B_error += num_error
            sum_B_word += len(seg_list)
    result = {
        'sum_A_filledpause': sum_A_filledpause,
        'sum_A_correction' : sum_A_correction,
        'sum_A_repeat'     : sum_A_repeat,
        'sum_A_error'      : sum_A_error,
        'sum_A_word'       : sum_A_word,
        'sum_B_filledpause': sum_B_filledpause,
        'sum_B_correction' : sum_B_correction,
        'sum_B_repeat'     : sum_B_repeat,
        'sum_B_error'      : sum_B_error,
        'sum_B_word'       : sum_B_word
    }
    return result
##读取数据：
final_list_test=pd.read_csv("data_final/2_final_list_test.csv") #大小（27,2）
final_list_train=pd.read_csv("data_final/2_final_list_train.csv") #大小（179,2）
#由于复赛数据噪声太大，故结合初赛
train1= pd.read_csv("data_final/1_preliminary_list_train.csv")#读取初赛的id
test1= pd.read_csv("data_final/1_preliminary_list_test.csv")
egemaps_pre=pd.read_csv("data_final/egemaps_final.csv") #大小（206,89）
print('训练集标签分布：\n',final_list_train['label'].value_counts())
#==================================================================================
##读取转写文本，即tsv文件夹下的文件，每一个文件都是对应一个人的，且每个文件行数不一定相等。
tsv_path_lists=os.listdir('data_final/tsv2/') #大小：206
tsv_feats=[] ##用于存放tsv特征
for path in tqdm(tsv_path_lists): ##遍历每个文件，提取特征
    z=pd.read_csv('data_final/tsv2/'+path,sep='\t')
    z['duration'] = z.end_time - z.start_time
    #读取文本的tfid特征
    vec = TfidfVectorizer(ngram_range=(1,1),min_df=1, max_df=0.99,use_idf=1,smooth_idf=1, sublinear_tf=10) #这里参数可以改
    trn_term_doc = vec.fit_transform(z['value'].values) 
    trn=pd.DataFrame(trn_term_doc.toarray())
    trn1=trn.describe()
    #结合baseline提供的文本特征
    #tsv1 = analysis_df(z)
    #df = pd.DataFrame([tsv1])
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
                      z['end_time-start_time'].median(),\
                      z['end_time-start_time'].skew(),\
                      z['end_time-start_time'].var(),\
                      z['jicha'].mean(),\
                      df1.sum_A_error[0]/df1.sum_A_word.values[0],\
                      df1.sum_B_filledpause[0]/df1.sum_B_word.values[0],\
                      np.max(trn1.loc['mean'],axis=0),\
                      np.max(trn1.loc['count'],axis=0),\
                      np.max(trn1.loc['min'],axis=0),\
                      np.max(trn1.loc['max'],axis=0),\
                      np.max(trn1.loc['count'],axis=0),\
                      np.max(trn1.loc['25%'],axis=0),\
                      np.max(trn1.loc['75%'],axis=0),\
                      z.shape[0]]) 
tsv_feats=pd.DataFrame(tsv_feats)
tsv_feats.columns=['uuid']+['tsv_feats{}'.format(i) for i in range(tsv_feats.shape[1]-1)]
#====================================================================================
##读取帧级别的Low-level descriptors (LLD)特征，即egemaps文件夹下的文件，每一个文件都是对应一个人的，且每个文件行数不一定相等。
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
final_list_train=final_list_train.merge(egemaps_pre,how='left',on=['uuid'])
final_list_train=final_list_train.merge(tsv_feats,how='left',on=['uuid'])
final_list_train=final_list_train.merge(egemaps_feats,how='left',on=['uuid'])

final_list_test=final_list_test.merge(egemaps_pre,how='left',on=['uuid'])
final_list_test=final_list_test.merge(tsv_feats,how='left',on=['uuid'])
final_list_test=final_list_test.merge(egemaps_feats,how='left',on=['uuid'])

#第一部分,结合初赛的结果
train1.drop(['label'],inplace=True,axis=1)
test1.drop(['label'],inplace=True,axis=1)
train1=train1.merge(final_list_train,how='left',on=['uuid'])
test1=test1.merge(final_list_test,how='left',on=['uuid'])
##标签映射：
label_dict1={'CTRL':0,'AD':1}
train1['label']=train1['label'].map(label_dict1)
sex_mapping1 = {'F':0, 'M':1}
train1['sex'] = train1['sex'].map(sex_mapping1)
test1['sex'] = test1['sex'].map(sex_mapping1)
ed
#特征选择
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
cols=[i for i in train1.columns if i not in ['uuid','label']]
sm = SelectFromModel(GradientBoostingClassifier(random_state=2019))
data_new= sm.fit_transform(train1.drop(['uuid','label'],axis=1).fillna(-99), train1['label'].values)
data_new1= sm.transform(test1.drop(['uuid','label'],axis=1).fillna(-99))
print(data_new.shape)
print([cols[i] for i in sm.get_support([0])])
#训练与预测
oof_xgb,prediction_xgb=\
      xgb_model(np.array(train1[[cols[i] for i in sm.get_support([0])]]),\
                train1['label'],\
                np.array(test1[[cols[i] for i in sm.get_support([0])]]),\
                0.01,5)
result1=[1 if i>0.5 else 0 for i in prediction_xgb]
test1['label']=result1
print(test1['label'].value_counts())
test1['label']=test1['label'].apply(lambda x:'AD' if x==1 else 'CTRL')      
#第二部分,主要是为了训练MCI
train2=final_list_train[~final_list_train.uuid.isin(train1.uuid.unique())].reset_index(drop=True)
#通过观察发现train2的label全为MCI
test2=final_list_test[~final_list_test.uuid.isin(test1.uuid.unique())].reset_index(drop=True)
label_dict2={'CTRL':0,'AD':2,'MCI':1}
sex_mapping2 = {'F':0, 'M':1}
test2['sex'] = test2['sex'].map(sex_mapping2)
final_list_train['label']=final_list_train['label'].map(label_dict2)
final_list_train['sex'] = final_list_train['sex'].map(sex_mapping2)
test2['sex'] = test2['sex'].map(sex_mapping2)

#===========================分割线=============================================
##模型训练预测
prediction_lgb,lgb=\
      lgb_model(final_list_train.drop(['uuid','label'],axis=1),\
                final_list_train['label'],\
                test2.drop(['uuid','label'],axis=1))
test2['label']=prediction_lgb
label_dict3={0:'CTRL',2:'AD',1:'MCI'}
test2['label']=test2['label'].map(label_dict3)
test3=pd.concat([test1,test2]).reset_index(drop=True)
final_list_test=final_list_test[['uuid']].merge(test3,how='left',on=['uuid'])
print(final_list_test['label'].value_counts())
preliminary_list_test[['uuid','label']].to_csv('submit.csv',index=None)


import matplotlib.pyplot as plt
labels = 'MCI','CTRL','AD'
sizes = [44.58,34.37,21.05]
explode=[0.1,0]
colors = [  'yellowgreen','gold','lightskyblue']
explode = (0,0,0.1)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.2f%%',\
        shadow=True,startangle=90)
plt.axis('equal')
plt.show()