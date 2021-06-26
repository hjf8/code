# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 13:29:41 2018

@author: 28770
"""

import pandas as pd

excelFile=r'ML_data2.xlsx'
train_df = pd.DataFrame(pd.read_excel(excelFile,sheet_name=0))  #读取指定路径的表格的sheet0为文件并转换到结构框格式
test_df= pd.DataFrame(pd.read_excel(excelFile,sheet_name=1)) #读取指定路径的表格的sheet1为文件并转换到结构框格式
ed
'''
#workClass_loss用于返回train_df中'workClass'这一列中的确实项，缺失数据处为True
workClass_loss=train_df['workClass'].isnull()  #.notnull()效果与其相反。
'''

'''
缺失值填充步骤：(使用缺失值上一行的数据填充缺失值处)
对train_df中的缺失值进行填充，其中.mode()是用这一列的众数填充，mean()使用列平均值填充。
其中，由于可能某一列有多个相同的众数，因此.mode()返回的是一个series,不像mean()一样返回
的是一个数值，因此，采用.mode()[0]自动将其填充为第一个众数。
'''
train_df_fill=train_df.fillna(method="ffill")
test_df_fill=test_df.fillna(method="ffill")

'''
删除重复的列信息
'''
train_df_fill=train_df_fill.drop(['education'],1)
test_df_fill=test_df_fill.drop(['education'],1)

'''
离散特征映射
'''
salary_mapping={'<=50K':0,'>50K':1}
train_df_fill['salary']=train_df_fill['salary'].map(salary_mapping)
test_df_fill['salary']=test_df_fill['salary'].map(salary_mapping)

Discrete_attribute=['workClass','education','marital_status','occupation'
                    'relationship','race','sex','native_country']

for attribute in Discrete_attribute:
    attribute_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill[attribute]))} 
    train_df_fill[attribute] = train_df_fill[attribute].map(attribute_mapping)  
    test_df_fill[attribute] = test_df_fill[attribute].map(attribute_mapping)

'''
workClass_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['workClass']))} 
train_df_fill['workClass'] = train_df_fill['workClass'].map(workClass_mapping)  
test_df_fill['workClass'] = test_df_fill['workClass'].map(workClass_mapping) 

education_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['education']))} 
train_df_fill['education'] = train_df_fill['education'].map(education_mapping)  
test_df_fill['education'] = test_df_fill['education'].map(education_mapping) 

marital_status_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['marital_status']))} 
train_df_fill['marital_status'] = train_df_fill['marital_status'].map(marital_status_mapping)  
test_df_fill['marital_status'] = test_df_fill['marital_status'].map(marital_status_mapping) 

occupation_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['occupation']))} 
train_df_fill['occupation'] = train_df_fill['occupation'].map(occupation_mapping)  
test_df_fill['occupation'] = test_df_fill['occupation'].map(occupation_mapping) 

relationship_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['relationship']))} 
train_df_fill['relationship'] = train_df_fill['relationship'].map(relationship_mapping)  
test_df_fill['relationship'] = test_df_fill['relationship'].map(relationship_mapping) 

race_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['race']))} 
train_df_fill['race'] = train_df_fill['race'].map(race_mapping)  
test_df_fill['race'] = test_df_fill['race'].map(race_mapping) 

sex_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['sex']))} 
train_df_fill['sex'] = train_df_fill['sex'].map(sex_mapping)  
test_df_fill['sex'] = test_df_fill['sex'].map(sex_mapping) 

native_country_mapping = {lab:idx for idx,lab in enumerate(set(train_df_fill['native_country']))} 
train_df_fill['native_country'] = train_df_fill['native_country'].map(native_country_mapping)  
test_df_fill['native_country'] = test_df_fill['native_country'].map(native_country_mapping)
'''


'''
#实现将多个DataFrame框架输出到一个excal中的不同的sheet中
filepath='H:/PYTHON模块程序/excel_change/ML_data2_trans.xlsx'
write=pd.ExcelWriter(filepath)
train_df_fill.to_excel(write,sheet_name='Sheet1',encoding='utf-8',index=False)
test_df_fill.to_excel(write,sheet_name='Sheet2',encoding='utf-8',index=False)
write.save()
'''

#实现将多个DataFrame框架输出到对应的CSV文件中
filepath_train='ML_data2_train.csv'
filepath_test='ML_data2_test.csv'

train_df_fill.to_csv(filepath_train,index=False)
test_df_fill.to_csv(filepath_test,index=False)








