# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

def Missingrate_Column(df, col):
    missing_records = df[col].map(lambda x: int(x!=x))
    return missing_records.mean()

def MakeupM_Missing(df,col, makeup_value):
    raw_values = list(df[col])
    missing_position = [i for i in range(len(raw_values)) if raw_values[i] != raw_values[i]]
    for i in missing_position:
        raw_values[i] = makeup_value
    return raw_values

def Outlier_Effect(df,col,target):
    p1, p3 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    d = p3 - p1
    upper, lower = p3 + 1.5 * d, p1 - 1.5 * d
    lower_sample, middle_sample, upper_sample = df[df[col]<lower], df[(df[col]>=lower)&(df[col]<=upper)], df[df[col]>upper]
    lower_pos, middle_pos, upper_pos = lower_sample[target].mean(), middle_sample[target].mean(), upper_sample[target].mean()
    lower_logodds, upper_logodds = np.log(lower_pos/middle_pos),np.log(upper_pos/middle_pos)
    return [lower_logodds, upper_logodds],lower,upper

def Detect_Outlier(x):
    p1,p3 = np.percentile(x,25), np.percentile(x,75)
    d = p3-p1
    upper, lower = p3+1.5*d, p1 - 1.5*d
    x2 = x.map(lambda x: min(max(x, lower), upper))
    return x2

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

def one_hot(label):
    y = np.zeros([len(label), 2])
    for index, value in enumerate(label):
        y[index][value] = 1
    return y

# 一、数据预处理
rawData = pd.read_csv('first_round_training_data.csv', header = 0)
rawData.drop(columns=['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10'], inplace=True)

d={ 'Fail':3, 'Pass':2, 'Good':1, 'Excellent':0 }
for l in list(d.keys()):
    rowIndex=rawData[rawData['Quality_label']==l].index
    rawData.loc[rowIndex, 'Quality_label'] = d[l]

trainData, valData = model_selection.train_test_split(rawData, test_size=0.3, random_state=42)
testData = pd.read_csv('first_round_testing_data.csv', header = 0)

testData.drop(columns=['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4'], inplace=True)

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
trainData.iloc[:,0:6] = trainData.iloc[:,0:6].apply(lambda x: np.log(x))
valData.iloc[:,0:6] = valData.iloc[:,0:6].apply(lambda x: np.log(x))
testData.iloc[:,1:7] = testData.iloc[:,1:7].apply(lambda x: np.log(x))

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
            max_features=4,
            oob_score=True,
            random_state=42,
            n_estimators=200,
            bootstrap=True,
            class_weight="balanced",
            max_depth=10,
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

## 预测方法1（测试集）
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

df['Excellent ratio']=df['Excellent ratio'].map(lambda x: round(x,2))
df['Good ratio']=df['Good ratio'].map(lambda x: round(x,2))
df['Pass ratio']=df['Pass ratio'].map(lambda x: round(x,2))
df['Fail ratio']=df['Fail ratio'].map(lambda x: round(x,2))

df.to_csv('sub091601.csv', index=False)    ## 线上：0.6451


## 预测方法2（测试集）
groupList = list(set(testData['Group']))
Fail_list, Pass_list, Good_list, Excellent_list=[],[],[],[]

x_test = np.array(testData)[:,1:testData.shape[1]]
pre_test = clf_RF.predict(x_test)

for i in groupList:
    Fail, Pass, Good, Excellent = 0, 0, 0, 0 
    for j in range(i*50, (i+1)*50):
        if(pre_test[j]==0):
            Fail+=1
        elif(pre_test[j]==1):
            Pass+=1
        elif(pre_test[j]==2):
            Good+=1
        else:
            Excellent+=1
    Fail_list.append(Fail)
    Pass_list.append(Pass)
    Good_list.append(Good)
    Excellent_list.append(Excellent)

Fail_list = list(np.array(Fail_list)/50)
Pass_list = list(np.array(Pass_list)/50)
Good_list = list(np.array(Good_list)/50)
Excellent_list = list(np.array(Excellent_list)/50)

result={'Group': groupList, 'Excellent ratio': Excellent_list, 'Good ratio': Good_list, 'Pass ratio': Pass_list, 'Fail ratio': Fail_list}
df=pd.DataFrame(result)

df.to_csv('RF0923.csv', index=False)   ## 线上：0.2907

## 三、xgboost
#X_dtrain, X_deval, y_dtrain, y_deval = cross_validation.train_test_split(X_train, y_train, random_state=1026, test_size=0.3)
#dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
#deval = xgb.DMatrix(X_deval, y_deval)
#watchlist = [(deval, 'eval')]
#params = {
#    'booster': 'gbtree',
#    'objective': 'reg:linear',
#    'subsample': 0.8,
#    'colsample_bytree': 0.85,
#    'eta': 0.05,
#    'max_depth': 7,
#    'seed': 2016,
#    'silent': 0,
#    'eval_metric': 'rmse'
#}
#clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
#pred = clf.predict(xgb.DMatrix(df_test))
#————————————————
#版权声明：本文为CSDN博主「wtq1993」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/wtq1993/article/details/51418958