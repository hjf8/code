# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:02:32 2019
@author: shaowu
"""
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
import time
t1=time.time()

#读入数据：
train=pd.read_csv(open('train.csv',encoding='utf-8'))
test=pd.read_csv(open('20190506_test.csv',encoding='utf-8'))

train['label']=train['label'].apply(lambda x:1 if x=='Positive' else 0)

#tf-idf特征：
column='review'
vec = TfidfVectorizer(ngram_range=(1,2),min_df=1, max_df=0.99,use_idf=1,smooth_idf=1, sublinear_tf=10) #这里参数可以改
trn_term_doc = vec.fit_transform(train[column])
xxx=trn_term_doc.toarray()
print(trn_term_doc.toarray())
test_term_doc = vec.transform(test[column])
label=train['label']
#x=np.asarray(train[column])
ed
#linesvc模型训练：
n_folds=10
print('LinerSVC stacking')
stack_train = np.zeros(train.shape[0])
stack_test = np.zeros(test.shape[0])

for i, (tr, va) in enumerate(StratifiedKFold(label, n_folds=n_folds, random_state=2018)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    lsvc = LinearSVC(random_state=2018)
    lsvc.fit(trn_term_doc[tr], label[tr])
    score_va = lsvc._predict_proba_lr(trn_term_doc[va])[:,1]
    score_te = lsvc._predict_proba_lr(test_term_doc)[:,1]
    print("per model roc_auc_score:", metrics.roc_auc_score(label[va],score_va))
    stack_train[va] += score_va
    stack_test += score_te
print("model roc_auc_score:", metrics.roc_auc_score(label,stack_train))
stack_test /= n_folds

##保存结果：
submit=test[['ID']]
submit['Pred']=stack_test
submit.to_csv('submit%s.csv'%time.time(), index=None, encoding='utf8')
