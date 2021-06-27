# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier


def Zero_Score_Normalization(series):
    p1, p3 = np.percentile(series, 25), np.percentile(series, 75)
    d = p3 - p1
    if d == 0:
        return -1
    upper, lower = p3 + 1.5 * d, p1 - 1.5 * d
    new_col = series.map(lambda x: min(max(x, lower),upper))
    mu,sigma = new_col.mean(), np.sqrt(new_col.var())
    new_var = new_col.map(lambda x: (x-mu)/sigma)
    return {'new_var':new_var,'lower':lower,'upper':upper,'mu':mu, 'sigma':sigma}

def Zero_Score_Normalization_2(df,col,percentiles=[1,99]):
    lower, upper = np.percentile(df[col], percentiles[0]), np.percentile(df[col], percentiles[1])
    d=upper-lower
    if d==0:
        return -1
    new_col = df[col].map(lambda x: min(max(x, lower),upper))
    #new_col = df[col]
    mu,sigma = new_col.mean(), np.sqrt(new_col.var())
    new_var = new_col.map(lambda x: (x-mu)/sigma)
    return {'new_var':new_var,'lower':lower,'upper':upper,'mu':mu, 'sigma':sigma}


# 一、数据预处理
rawData = pd.read_csv('first_round_training_data.csv', header = 0)
rawData.drop(columns=['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10'], inplace=True)

d={ 'Fail':3, 'Pass':2, 'Good':1, 'Excellent':0 }
for l in list(d.keys()):
    rowIndex=rawData[rawData['Quality_label']==l].index
    rawData.loc[rowIndex, 'Quality_label'] = d[l]

trainData, valData = model_selection.train_test_split(rawData, test_size=0.3, random_state=42)
testData = pd.read_csv('first_round_testing_data.csv', header = 0)

# 删除异常样本
#trainData=trainData[trainData['Parameter1']<=0.5*pow(10,8)]
#trainData=trainData[trainData['Parameter2']<=2*pow(10,6)]
#trainData=trainData[trainData['Parameter4']<=3.7*pow(10,8)]
#trainData=trainData[trainData['Parameter5']<=70]
#trainData=trainData[trainData['Parameter7']<=2.4*pow(10,4)]
#trainData=trainData[trainData['Parameter8']<=1.0*pow(10,4)]
#trainData=trainData[trainData['Parameter9']<=1.0*pow(10,8)]

#valData=valData[valData['Parameter4']<=3.7*pow(10,8)]
#valData=valData[valData['Parameter5']<=70]
#valData=valData[valData['Parameter7']<=2.4*pow(10,4)]

#testData=testData[testData['Parameter1']<=0.5*pow(10,8)]
#testData=testData[testData['Parameter2']<=2*pow(10,6)]
#testData=testData[testData['Parameter4']<=3.7*pow(10,8)]
#testData=testData[testData['Parameter5']<=70]
#testData=testData[testData['Parameter7']<=2.4*pow(10,4)]
#testData=testData[testData['Parameter8']<=1.0*pow(10,4)]
#testData=testData[testData['Parameter9']<=1.0*pow(10,8)]

# 数据做log变换
#对前4个变量做log变换
#trainData.iloc[:,[0,1,2,3]]=trainData.iloc[:,[0,1,2,3]].apply(lambda x: np.log(x))
#valData.iloc[:,[0,1,2,3]]=valData.iloc[:,[0,1,2,3]].apply(lambda x: np.log(x))
#testData.iloc[:,[1,2,3,4]]=testData.iloc[:,[1,2,3,4]].apply(lambda x: np.log(x))

#对10个变量做log变换
trainData.iloc[:,0:10] = trainData.iloc[:,0:10].apply(lambda x: np.log(x))
valData.iloc[:,0:10] = valData.iloc[:,0:10].apply(lambda x: np.log(x))
testData.iloc[:,1:11] = testData.iloc[:,1:11].apply(lambda x: np.log(x))

##离散化函数
#def fun(x, dic):
# if x in list(dic.keys()):
#     return dic[x]
# else:
#     return 0
#
## 有序离散型变量，用1，2，...,n数值离散化
##params = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
#params = ['Parameter8', 'Parameter9']

## 保留离散化前的原变量
#trainData['Parameter11']=trainData['Parameter8']
#trainData['Parameter12']=trainData['Parameter9']
#valData['Parameter11']=valData['Parameter8']
#valData['Parameter12']=valData['Parameter9']
#testData['Parameter11']=testData['Parameter8']
#testData['Parameter12']=testData['Parameter9']

## 把'Quality_label'放在最后一列，并命名为 'label'
#trainData['label']=trainData['Quality_label']
#valData['label']=valData['Quality_label']
#trainData.drop(columns=['Quality_label'], inplace=True)
#valData.drop(columns=['Quality_label'], inplace=True)

## 对变量离散化
#for p in params:
#    dic = dict(trainData[p].value_counts())
#    dic_keys = np.sort(list(dic.keys()))
#    for i in range(1,len(dic_keys)+1):
#        dic[dic_keys[i-1]] = i
#    trainData[p]= trainData[p].apply(lambda x: fun(x, dic))
#    valData[p]= valData[p].apply(lambda x: fun(x, dic))
#    testData[p]= testData[p].apply(lambda x: fun(x, dic))

### 归一化, 1%分位数和99%分位数
lower, upper, mu, sigma = {}, {}, {}, {}
cols = list(trainData.columns)
#cols.remove('label')
cols.remove('Quality_label')
deleted_cols = []

for col in cols:
    zero_score = Zero_Score_Normalization_2(trainData, col)
    if zero_score == -1:
        deleted_cols.append(col)
        continue
    trainData[col] = zero_score['new_var']
    lower[col], upper[col], mu[col], sigma[col] =zero_score['lower'], zero_score['upper'], zero_score['mu'], zero_score['sigma']

    #new_col = valData[col]
    new_col = valData[col].map(lambda x: min(max(x, lower[col]),upper[col]))
    new_var = new_col.map(lambda x: (x-mu[col])/sigma[col])
    valData[col] = new_var
    
    #new_col = testData[col]
    new_col = testData[col].map(lambda x: min(max(x, lower[col]),upper[col]))
    new_var = new_col.map(lambda x: (x-mu[col])/sigma[col])
    testData[col] = new_var
    
for col in deleted_cols:
    del trainData[col]
    del valData[col]
    del testData[col]


# 二、构建随机森林模型
x_train = np.array(trainData)[:,0:(trainData.shape[1]-1)]  
y_train = np.array(trainData)[:, trainData.shape[1]-1]

x_val = np.array(valData)[:,0:(valData.shape[1]-1)]  
y_val = np.array(valData)[:, valData.shape[1]-1]


# 模型：随机森林  max_depth=6, max_depth如果不设置，过拟合
# max_features: If int,特征数; If float, 特征总数的百分比; If “auto”, sqrt(n_features); If “sqrt”, sqrt(n_features); If “log2”, log2(n_features); If None, 全部特征数
clf_RF = RandomForestClassifier(
            max_features=9,
            oob_score=True,
            random_state=42,
            n_estimators=500,
            bootstrap=True,
            class_weight="balanced",
            max_depth=17,
            n_jobs=10)
clf_RF.fit(x_train, y_train)

# 训练集
pre_train = clf_RF.predict(x_train)
preProba_train = clf_RF.predict_proba(x_train)
acc_train = 1.0 * sum(np.equal(pre_train, y_train)) / len(y_train)

N1=int(x_train.shape[0]/50)  #84组
preResult_train = []  ## 预测每组在4类的概率
trueResult_train = []  ## 真实概率
n_samples = 50
for i in range(N1):
    preResult_train.append(list(np.mean(preProba_train[n_samples*i:(n_samples*(i+1))], axis=0)))
    tmp = y_train[n_samples*i:(n_samples*(i+1))]
    trueResult_train.append([sum(tmp==0)/n_samples, sum(tmp==1)/n_samples, sum(tmp==2)/n_samples, sum(tmp==3)/n_samples])

preResult_train = np.array(preResult_train)    # 84*4
trueResult_train = np.array(trueResult_train)  # 84*4

MAE_train = np.sum(abs(preResult_train-trueResult_train))/(preResult_train.shape[0]*preResult_train.shape[1])
Score_train = 1.0/(1.0+10*MAE_train)
print('Acc_train: ', acc_train)
print('MAE_train: ', MAE_train)
print('Score_train: ', Score_train)

# 验证集
pre_val = clf_RF.predict(x_val)
preProba_val = clf_RF.predict_proba(x_val)
acc_val = 1.0 * sum(np.equal(pre_val, y_val)) / len(y_val)

N2=int(x_val.shape[0]/50)  ## 36组
preResult_val = []
trueResult_val = []
for i in range(N2):
    preResult_val.append(list(np.mean(preProba_val[n_samples*i:(n_samples*(i+1))], axis=0)))
    tmp = y_val[n_samples*i:(n_samples*(i+1))]
    trueResult_val.append([sum(tmp==0)/n_samples, sum(tmp==1)/n_samples, sum(tmp==2)/n_samples, sum(tmp==3)/n_samples])

preResult_val = np.array(preResult_val)    ## 36*4
trueResult_val = np.array(trueResult_val)  ## 36*4

MAE_val = np.sum(abs(preResult_val-trueResult_val))/(preResult_val.shape[0]*preResult_val.shape[1])
Score_val = 1.0/(1.0+10*MAE_val)
print('Acc_val: ', acc_val)
print('MAE_val: ', MAE_val)
print('Score_val: ', Score_val)

## 预测（测试集）
groupList = list(set(testData['Group']))
dic = dict(testData['Group'].value_counts())

x_test = np.array(testData)[:,1:testData.shape[1]]
pre_test = clf_RF.predict_proba(x_test)
#求group下同一组的平均值
result = []
start=0
for i in groupList:
    result.append(list(np.mean(pre_test[start:(start+dic[i])], axis=0)))
    start+= dic[i]
result=np.array(result)

result={'Group': groupList, 'Excellent ratio': list(result[:,0]), 'Good ratio': list(result[:,1]), 'Pass ratio': list(result[:,2]), 'Fail ratio': list(result[:,3])}
df=pd.DataFrame(result)

df.to_csv('submit0909.RF.01.csv', index=False)