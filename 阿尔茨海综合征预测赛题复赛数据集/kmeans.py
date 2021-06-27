# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 20:48:52 2019

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
    返回:
        sum_filledpause : 语气词（有声停顿）的个数之和
        sum_correction : 修正的次数之和
        sum_repeat : 重复的次数之和
        sum_error : 语法错误的次数之和
    '''
    # process one tsv
    # tsv_df['text_seg'] = None
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
        #print(type(speaker))
        if type(speaker) is not str:
            print('warning %s:%d line: speaker:%s' % (indx, str(speaker)))
            continue
        if len(value)>7:
            ori_text, num_filledpause, num_correction, num_repeat, num_error = get_origin_text(value)
            #print(ori_text)
            if ori_text == '':
                continue
            seg_list = jieba.cut(ori_text, cut_all=False, HMM=True)
            
            seg_list = list(seg_list)
            #print(seg_list)
            if speaker.strip() == '<A>':
                sum_A_filledpause += num_filledpause
                sum_A_correction += num_correction
                sum_A_repeat += num_repeat
                sum_A_error += num_error
                sum_A_word += len(seg_list)
            # if outfile_path:
            #     out_f.write(' '.join(seg_list)+'\n')
            elif speaker.strip() == '<B>':
                sum_B_filledpause += num_filledpause
                sum_B_correction += num_correction
                sum_B_repeat += num_repeat
                sum_B_error += num_error
                sum_B_word += len(seg_list)
            # tsv_df.loc[indx, 'text_seg'] = '/'.join(seg_list)
    # tsv_df.to_csv(outtsv_path, encoding='utf-8', sep='\t', index=Flase)
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
                      z['end_time-start_time'].median(),\
                      z['end_time-start_time'].skew(),\
                      z['end_time-start_time'].var(),\
                      z['jicha'].mean(),\
                      df['total_duration'].values[0],\
                      df['A_speak_num'].values[0],\
                      df['A_speak_duration_mean'].values[0],\
                      df['A_speak_duration_std'].values[0],\
                      df['A_speak_duration_median'].values[0],\
                      df['B_speak_num'].values[0],\
                      df['B_speak_duration_mean'].values[0],\
                      df['B_speak_duration_std'].values[0],\
                      df['B_speak_duration_median'].values[0],\
                      df['B_speak_duration_skew'].values[0],\
                      df['A_speak_duration_proportion'].values[0],\
                      df['B_speak_duration_proportion'].values[0],\
                      df['slience_duration_proportion'].values[0],\
                      df1.sum_A_filledpause[0]/df1.sum_A_word.values[0],\
                      df1.sum_A_correction[0]/df1.sum_A_word.values[0],\
                      df1.sum_A_repeat[0]/df1.sum_A_word.values[0],\
                      df1.sum_A_error[0]/df1.sum_A_word.values[0],\
                      df1.sum_B_filledpause[0]/df1.sum_B_word.values[0],\
                      df1.sum_B_correction[0]/df1.sum_B_word.values[0],\
                      df1.sum_B_repeat[0]/df1.sum_B_word.values[0],\
                      df1.sum_B_error[0]/df1.sum_B_word.values[0],\
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
preliminary_list_train.fillna(0,inplace=True)
print(preliminary_list_train.shape)
ed
data = preliminary_list_train.drop(['uuid','label'],axis=1)
data1 = preliminary_list_test.drop(['uuid','label'],axis=1)
from sklearn.cluster import KMeans #导入K均值聚类算法
k = 3                     #需要进行的聚类类别数
iteration = 800             #聚类最大循环数
kmodel = KMeans(n_clusters = k, n_jobs = 4) #n_jobs是并行数，一般等于CPU数较好
kmodel.fit(data) #训练模型
r1 = pd.Series(kmodel.labels_).value_counts()  #统计各个类别的数目
r2 = pd.DataFrame(kmodel.cluster_centers_)     #找出聚类中心
r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r.columns = list(data.columns) + [u'类别数目'] #重命名表头
print(r)

kmodel2 = KMeans(n_clusters = k, n_jobs = 4) #n_jobs是并行数，一般等于CPU数较好
kmodel2.fit(data1) 
r11 = pd.Series(kmodel2.labels_).value_counts()  #统计各个类别的数目
r21= pd.DataFrame(kmodel2.cluster_centers_)     #找出聚类中心
r1 = pd.concat([r21, r11], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r1.columns = list(data1.columns) + [u'类别数目'] #重命名表头
print(r1)
ed
test1= pd.read_csv("data_final/1_preliminary_list_test.csv")
test1.drop(['label'],inplace=True,axis=1)
data11=test1.merge(preliminary_list_test,how='left',on=['uuid'])
data2 = data11.drop(['uuid','label'],axis=1)
kmodel3 = KMeans(n_clusters = 2, n_jobs = 4) #n_jobs是并行数，一般等于CPU数较好
kmodel3.fit(data2) #训练模型
r13 = pd.Series(kmodel3.labels_).value_counts()  #统计各个类别的数目
r3 = pd.DataFrame(kmodel3.cluster_centers_)     #找出聚类中心
r31 = pd.concat([r3, r13], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
r31.columns = list(data2.columns) + [u'类别数目'] #重命名表头
print(r31)
r32 = pd.concat([data2, pd.Series(kmodel3.labels_, index = data2.index)], axis = 1)  #详细输出每个样本对应的类别
data11['julei']=r32[0]
r1 = pd.concat([data2, pd.Series(kmodel3.labels_, index = data2.index)], axis = 1)  #详细输出每个样本对应的类别




r = pd.concat([data, pd.Series(kmodel.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
preliminary_list_train['julei']=r[0]
r1 = pd.concat([data1, pd.Series(kmodel2.labels_, index = data1.index)], axis = 1)  #详细输出每个样本对应的类别
preliminary_list_test['julei']=r1[0]

train1=preliminary_list_train[preliminary_list_train.julei==0].reset_index(drop=True)
test1=preliminary_list_test[preliminary_list_test.julei==0].reset_index(drop=True)
prediction_lgb,lgb11=\
      lgb_model(train1.drop(['uuid','label'],axis=1),\
                train1['label'],\
                test1.drop(['uuid','label'],axis=1))
test1['label']=prediction_lgb
print(test1['label'].value_counts())
label_dict1={0:'CTRL',2:'AD',1:'MCI'}
test1['label']=test1['label'].map(label_dict1)



print(preliminary_list_test['label'].value_counts())
label_dict1={0:'CTRL',2:'AD',1:'MCI'}
preliminary_list_test['label']=preliminary_list_test['label'].map(label_dict1)
preliminary_list_test[['uuid','label']].to_csv('submit.csv',index=None)