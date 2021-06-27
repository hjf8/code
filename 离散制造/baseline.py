import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def custom_fun(preds, labels):
    temp=abs( preds- labels)
    res=np.sum(temp)/(len(temp)*4)
    result=1/(1+10*res)
    return result

def cuntom_loss(preds, train_data):
    labels = np.array(train_data.get_label())
    preds = np.reshape(preds, (len(labels), -1), order='F')
    labels = OneHotEncoder(n_values=4, sparse=False).fit_transform(labels.reshape(-1, 1))
    res = custom_fun(preds, labels)
    return 'custom_mutli_loss', res, True

def Lgb_Classifier(train_x, train_y,test_x,test_y):
    import lightgbm as lgb
    lgb_trn = lgb.Dataset(
    data=train_x,
    label=train_y,free_raw_data=True)
    lgb_val = lgb.Dataset(data=test_x,label=test_y,free_raw_data=True)
    params_lgb = { 'boosting_type': 'gbdt','objective': 'multiclass','num_class': 4, 'metric': 'multi_error','num_leaves': 300, 'learning_rate': 0.01, 'num_threads':-1}
    fit_params_lgb = {'num_boost_round': 2500, 'verbose_eval':50,'early_stopping_rounds':1500}
    lgb_reg = lgb.train(params=params_lgb, train_set=lgb_trn,feval=cuntom_loss,**fit_params_lgb, valid_sets=[lgb_trn, lgb_val])
    return lgb_reg

def Cat_Classifier(train_x, train_y,test_x,test_y,cat_features):
    from catboost import Pool, CatBoostClassifier
    cat_trn = Pool(data=train_x,label=train_y, cat_features=cat_features )
    cat_val = Pool(data=test_x,label=test_y,cat_features=cat_features)
    model=CatBoostClassifier(max_depth=7,learning_rate=0.01,verbose=300,n_estimators=5000,random_state=2020)
    cat_reg=model.fit(cat_trn,eval_set=cat_val)
    return cat_reg

def model_cv(train,test,esti,feature_names,cat_features):
    oof = np.zeros((train.shape[0],4))
    cv_model=[]
    skf = StratifiedKFold(n_splits=nfold, random_state=2020, shuffle=True)
    for fold_id, (train_index, test_index) in enumerate(skf.split(train[feature_names],train[ycol])):
        print(f'\nFold_{fold_id} Training ================================\n')
        reg = { 'CAT':Cat_Classifier,'LGB':Lgb_Classifier}
        train_x, train_y = train.loc[train_index, feature_names], train.loc[train_index,ycol]
        test_x, test_y = train.loc[test_index, feature_names], train.loc[test_index, ycol]
        if esti=='CAT' :
            model= reg[esti](train_x, train_y,test_x,test_y,cat_features)
            oof[test_index] =model.predict_proba(train.iloc[test_index][feature_names])
            cv_model.append(model)
            if len(test):
                test.loc[:,submit.columns[1:]] += model.predict_proba(test[feature_names]) / nfold
        elif esti=='LGB' :
            model= reg[esti](train_x, train_y,test_x,test_y)
            oof[test_index] = model.predict(test_x,num_iteration=model.best_iteration)
            cv_model.append(model)
            if len(test):
                test.loc[:,submit.columns[1:]] += model.predict(test[feature_names],num_iteration=model.best_iteration) / nfold
    return oof,test,cv_model

ycol='result'
nfold=10
train = pd.read_csv('first_round_training_data.csv')
test = pd.read_csv('first_round_testing_data.csv')
submit = pd.read_csv('submit_example.csv')
for predcol in submit.columns[1:]:
    test[predcol] = 0
train[ycol] = train.Quality_label.map({'Excellent':0,'Good':1,'Pass':2,'Fail':3})
dropcols=['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Attribute1', 'Attribute2','Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8','Attribute9', 'Attribute10', 'Quality_label', ycol]
feature_names=list(filter(lambda x:x not in dropcols,train.columns))
print(feature_names)
cat_features=[]
esti='CAT'
oof,test,cv_model=model_cv(train,test,esti,feature_names,cat_features)
submit = test.groupby(['Group'],as_index=False)[submit.columns[1:]].mean()
submit.to_csv('baseline02_0923_02.csv',index=False)