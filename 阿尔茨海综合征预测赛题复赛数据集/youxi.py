# -*- coding: utf-8 -*-
from selenium import webdriver
from tqdm import *
base_url = 'https://android.myapp.com/myapp/detail.htm?apkName='


'''
    1.需要安装selenium库
    2.需要下载chromedriver，并放在相应目录
    3.要想获取文件中（final_apptype_train）所有APP的详细信息：
        3.1 读取文件内容
        3.2 循环获取每个app的name，如zhongxinjiantou.szkingdom.android.newphone
        3.3 将name传入download函数中，即可获取每个应用的详细信息
'''


def download(app_name):
    browser = webdriver.Chrome()
    browser.get(base_url+app_name)
    detail = browser.find_element_by_css_selector(".det-app-data-info")
    #print(detail.text)
    x=detail.text
    browser.close()
    return x

shuju = []
count = 0
import pandas as pd
train= pd.read_csv("final_apptype_train.dat",header=None,encoding='utf8',delimiter='\t')
train.columns=['id','conment','label']
for indx in tqdm(range(len(train))):
    try:
        shuju.append(download(train.loc[indx,'conment']))
    except:
        shuju.append(0)

    
    
    


 
    