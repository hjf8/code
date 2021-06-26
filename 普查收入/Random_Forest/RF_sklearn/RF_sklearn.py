# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:04:38 2018

@author: 28770
"""
import csv
from sklearn.ensemble import RandomForestClassifier

def loadCSV(filename):  #读取csv文件
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

if __name__=='__main__':
    train_dataSet=loadCSV('ML_data2_train.csv')
    test_dataSet=loadCSV('ML_data2_test.csv')
    train_set=[]
    test_set=[]
    for row in test_dataSet:
        row_copy=list(row)
        row_copy.pop(14)
        test_set.append(row_copy) #测试集的属性信息
    for row in train_dataSet:
        row_copy=list(row)
        row_copy.pop(14)
        train_set.append(row_copy) #训练集的属性信息
    train_label=[row[-1] for row in train_dataSet] #训练集标签信息
    test_label=[row[-1] for row in test_dataSet] #测试集标签信息
    clf = RandomForestClassifier(n_estimators = 80,max_depth=30) #决策树数量80，最大树深30
    RF = clf.fit(train_set , train_label) #构建随机森林模型
    accur = clf.score(test_set , test_label) #计算预测准确率
    print ('accur:%s'% accur)







