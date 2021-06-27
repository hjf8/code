# -*- coding: utf-8 -*-
from selenium import webdriver
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import *

'''
    1.需要安装selenium库
    2.需要下载chromedriver，并放在相应目录
    3.要想获取文件中（final_apptype_train）所有APP的详细信息：
        3.1 读取文件内容
        3.2 循环获取每个app的name，如zhongxinjiantou.szkingdom.android.newphone
        3.3 将name传入download函数中，即可获取每个应用的详细信息
'''

base_url = 'https://android.myapp.com/myapp/detail.htm?apkName='
def download(app_name):
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    browser = webdriver.Chrome(options=option)
    browser.get(base_url+app_name)
    detail = ""

    try:
        detail = browser.find_element_by_css_selector(".det-app-data-info")
        print("已获取此应用信息：", app_name)
    except Exception as e:
        print("此应用不存在:", app_name)
    if detail:
        x=detail.text
        return x
    else:return 0
        #with open("result.txt","a") as file:
            # print(detail.text)
            #file.write(detail.text+"###$$$%%%\n")
    browser.close()
if __name__ == "__main__":
    train= pd.read_csv("final_apptype_train.dat",header=None,encoding='utf8',delimiter='\t')
    train.columns=['id','conment','label']

    #test=pd.read_csv("appname_package.dat",header=None,encoding='utf8',delimiter='\t')
    feats=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        train_files=list(train.conment[4180:10000])
        
        for path,res in tqdm(zip(train_files,executor.map(download,train_files))):
            feats.append(res)
'''

from selenium import webdriver
from multiprocessing import Pool

base_url = 'https://android.myapp.com/myapp/detail.htm?apkName='


# 获取某个app的信息

def download(app_name):
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    browser = webdriver.Chrome(options=option)
    browser.get(base_url+app_name)
    detail = ""

    try:
        detail = browser.find_element_by_css_selector(".det-app-data-info")
        print("已获取此应用信息：", app_name)
    except Exception as e:
        print("此应用不存在:", app_name)
    #if detail:
        #x=detail.text
    #else:x = 0
    if detail:
        x=detail.text
        shuju.append(x)
        #with open("result.txt","a") as file:
            # print(detail.text)
            #file.write(detail.text+"###$$$%%%\n")
    browser.close()


# 获取所有app name
def get_apps():
    train= pd.read_csv("final_apptype_train.dat",header=None,encoding='utf8',delimiter='\t')
    train.columns=['id','conment','label']
    app_names = list(train['conment'][:10])
    #with open("final_apptype_train.txt", "r") as f:
        #results = f.read().split("\n")
        #app_names = list()
        #for result in results:
         #   result = result.split("\t")[1]
         #   app_names.append(result)
    return app_names


def main():
    po = Pool(4)
    app_names = get_apps()
    for app_name in app_names:
        po.apply_async(download, (app_name,))
    po.close()
    po.join()


if __name__ == '__main__':
    #feats=[]
    shuju = []
    main()
'''
#xx=pd.DataFrame(feats)
#xx.to_csv('3540_4180.csv',index=None,encoding='utf_8_sig')