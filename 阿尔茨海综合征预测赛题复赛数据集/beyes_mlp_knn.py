# -*- coding: utf-8 -*-
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors,tree
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
def xgb_model(X_train,y_train,X_test,y_test,test):
    ##划分训练集与验证集， 8:2划分
 #   X, val_X, y, val_y = train_test_split(train_x,train_y,test_size=0.3,random_state=1,stratify=train_y)## 这里保证分割后y的比例分布与原数据一致  
 #   X_train = X  
 #   y_train = y  
 #   X_test = val_X  
 #   y_test = val_y
 #   test.columns=X.columns
    dtrain=xgb.DMatrix(X_train,y_train)
    dval=xgb.DMatrix(X_test,y_test)
    dtest=xgb.DMatrix(X_test)
    test=xgb.DMatrix(test)
    num_rounds=2000  #迭代次数
    params={'booster':'gbtree',
            'eta':0.8,
            'max_depth':5,
            'objective':'multi:softmax',#'binary:logistic',
            'eval_metric': 'merror',
            'num_class':10,
         #   'min_child_weight':2,
          #  'gamma':0.2,
   #         'subsample':0.9,
    #        'colsample_bytree':0.9,
       #     'nthread':4,
         #   'scale_pos_weight':1,
            'random_seed':2018 
            }
    watchlist = [(dtrain,'train'),(dval,'val')]
    ####模型训练：
    xgb_model=xgb.train(params,dtrain,num_rounds,watchlist,early_stopping_rounds=10)
    ###模型验证
    pre_y=xgb_model.predict(dtest,ntree_limit=xgb_model.best_ntree_limit)
    print(confusion_matrix(y_test,pre_y))
    #计算准确率:
    print("xgb model accuracy:", metrics.accuracy_score(y_test,pre_y))
    print(classification_report(y_test,pre_y))
    
    ####模型预测：
    
    res=xgb_model.predict(test,ntree_limit=xgb_model.best_ntree_limit)
    print(Counter(res))
    return res


###定义读入数据函数
def read_data():
    train = pd.read_csv('training.csv',header=None)
    test = pd.read_csv('preliminary-testing.csv',header=None)
    train.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5','ini_label']
    
    train_y=train['ini_label']
    train_x=train.drop(['ini_label'],axis=1)
    test.columns=train_x.columns
    return train_x,train_y,test

def split_data(train_x,train_y,radio):
    ###划分数据：返回的X为训练数据，val为验证数据，y为训练数据中的标签，val_y为验证数据中的标签
    ###test_size为划分的比例，
    X, val, y, val_y = train_test_split(train_x,train_y,test_size=radio,random_state=1,stratify=train_y)
    return X, val, y, val_y

###建立bayes模型（三种朴素贝叶斯）：
def bayes_model(x, val, y, val_y,test):
    ##默认参数
    gs= GaussianNB()#这里使用默认的参数
    m=MultinomialNB(alpha=0.5)
    b=BernoulliNB()
    #训练模型
    gs.fit(x, y)
    m.fit(x, y)
    b.fit(x, y)
    #验证：
    pre_y = gs.predict(val)
    pre_y2 = m.predict(val)
    pre_y3 = b.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("Gaussian-Bayes model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(confusion_matrix(val_y,pre_y2))
    #计算准确率:
    print("Mult-Bayes model accuracy:", metrics.accuracy_score(val_y, pre_y2))
    print(confusion_matrix(val_y,pre_y3))
    #计算准确率:
    print("BernoulliNB model accuracy:", metrics.accuracy_score(val_y, pre_y3))
    
    ##预测：
    res=gs.predict(test) #用gs模型预测
    return res


###建立mlp模型（多层感知机）：
def mlp_model(x, val, y, val_y,test):
    ##设定参数：
    mlp = MLPClassifier(hidden_layer_sizes=
                     #   (85,85,85),#97.8
                        (1360,170,85), #0.9986
                       # (1360,85,85),#0.9912
                       # (340,255,85),#99.840
                       # (340,85,85), #0.99780隐含层大小
                       # (255,170,85), #0.9982隐含层大小
                        max_iter=1000, #迭代次数
                        activation='relu', #激活函数
                        alpha=0.0001, 
                        batch_size='auto',
                        beta_1=0.5,#0.9,
                        beta_2=0.999,
                        early_stopping=False, 
                        epsilon=1e-08,
                        learning_rate='constant',
                        learning_rate_init=0.001, #学习率 
                        momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=2018,
                        shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                        verbose=False, warm_start=False)
    #模型训练：
    mlp.fit(x,y)
    #验证：
    pre_y = mlp.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("mlp model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    
    del x,y,val,val_y
    ##预测：
    
    res=mlp.predict(test)
    return res

###建立knn模型（K-近邻算法）：
def knn_model(x, val, y, val_y,test):
    """
    ###数据标准化：
    scaler = StandardScaler()
    scaler.fit(x)
    x= scaler.transform(x)
    val = scaler.transform(val)
    """
    #参数设定：
    clf=neighbors.KNeighborsClassifier(algorithm='auto',leaf_size=10, metric='minkowski',
           metric_params=None, n_jobs=4, n_neighbors=10, p=85,
           weights='uniform') ##n_jobs为进程数，p为特征数
    #训练模型：
    clf.fit(x,y)
    #预测：
    pre_y= clf.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("KNN model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    
    ##预测：
    res=clf.predict(test)
    return res

###建立决策树模型
def tree_model(x, val, y, val_y,test):
    model=tree.DecisionTreeClassifier(criterion='entropy', splitter='best',
                                      max_depth=8, min_samples_split=2,
                                      min_samples_leaf =2, min_weight_fraction_leaf=0.,
                                      max_features=None, random_state=2018, 
                                      max_leaf_nodes=None,class_weight=None, presort=False)
    #criterion='entropy'信息增益；'gini'
    ##模型训练：
    model.fit(x,y)
    ##模型验证：
    pre_y=model.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("DecisionTreeClassifier model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    
    ##预测：
    res=model.predict(test)
    return res
###建立随机森林模型    
def rf_model(x, val, y, val_y,test):
    #默认参数：
    model=RandomForestClassifier(random_state=2018)
    #训练模型:
    model.fit(x,y)
    #模型验证：
    pre_y=model.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("RandomForestClassifier model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))

    ##预测：
    res=model.predict(test)
    return res
###建立    
def ext_model(x, val, y, val_y,test):
    model=ExtraTreesClassifier()
    #模型训练：
    model.fit(x,y)
    #验证：
    pre_y=model.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("ExtraTreesClassifier model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    
    ##预测：
    res=model.predict(test)
    return res

def ada_model(x, val, y, val_y,test):
    model=AdaBoostClassifier(
            algorithm='SAMME',n_estimators=20,learning_rate=0.1,)
    ##模型训练：
    model.fit(x,y)
    #验证：
    pre_y=model.predict(val)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("AdaBoostClassifier model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))

    ##预测：
    res=model.predict(test)
    return res
def lf_model(x, val, y, val_y,test):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=2018)
    lr.fit(x, y)
    pre=lr.predict_proba(val)
    pre_y=np.argmax(pre,axis=1)
    print(confusion_matrix(val_y,pre_y))
    #计算准确率:
    print("AdaBoostClassifier model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    
    ##预测：
    res=lr.predict_proba(test)
    res=np.argmax(res,axis=1)
    
    print(Counter(res))
    return res
    


def candidate_set(data):
    #cols=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5','ini_label']
    newdata=pd.DataFrame()
    m=pd.DataFrame()
    m['flag']=[0,1,2,3,4,5,6,7,8,9]
    m['label']=[0,0,0,0,0,0,0,0,0,0]
    n=data.shape[0]
    for i in range(n):
        user=np.tile(data.ix[i],(10,1))
        user=pd.DataFrame(user)
        user.columns=data.columns
        user=pd.concat([user,m],axis=1)
        for j in range(10):
            if user.ix[j,'ini_label']==user.ix[j,'flag']:
                user.ix[j,'label']=1
                break
        user=user.drop(['ini_label'],axis=1)
        newdata=pd.concat([newdata,user])
    return newdata
if __name__ == "__main__":
    #读入数据：
    train = pd.read_csv('training.csv',header=None)
    test = pd.read_csv('preliminary-testing.csv',header=None)  
    train.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5','ini_label']
    test.columns=['ziduan1','num1','ziduan2','num2','ziduan3','num3','ziduan4','num4','ziduan5','num5']
    

    
    ##删除重复行：
    train=train.drop_duplicates().reset_index(drop=True)
    ###全部类别进行排列组合：
    # train=gather_data(train_x)
    
    ##构造候选集：
#    train=candidate_set(train)
   

    
    #train.to_csv('candidate_train_set.csv',index=None)
    #train = pd.read_csv('candidate_train_set.csv')
    
    train_y=train['ini_label']
    train_x=train.drop(['ini_label'],axis=1)
    
    
    ########################################################################################
    x, val, y, val_y= train_test_split(train_x,train_y,test_size=0.4,random_state=2018,stratify=train_y)
    
    #########################################################################################
    ##flag特征不用编码：
#    x_flag=x['flag'].reset_index(drop=True)
#    val_flag=val['flag'].reset_index(drop=True)
#    x=x.drop(['flag'],axis=1)
#    val=val.drop(['flag'],axis=1)

    
    x=x.toarray()
    val=val.toarray()
    test=test.toarray()
#    val=pd.DataFrame(val)
#    x=pd.DataFrame(x)
#    test=pd.DataFrame(test)
#    val['flag']=val_flag
#    x['flag']=x_flag
    
    #del train_x,train_y
    #################################数据归一化###############################################
    
    ###数据标准化：
#    scaler = StandardScaler()
#    scaler.fit(x)
#    x= scaler.transform(x)
#    val = scaler.transform(val)

    ################################开始训练和验证和预测#######################################
    
 #   res=xgb_model(x,y,val,val_y,test)

    ################################保存结果##################################################
 #   np.savetxt("dsjyycxds_preliminary.txt",res1.astype(int),fmt="%d"+'\r')