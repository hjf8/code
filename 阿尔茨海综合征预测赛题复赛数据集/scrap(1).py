# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from lxml import etree
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import *
#options = webdriver.ChromeOptions()
#options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
#chrome_driver_binary = "chromedriver"
#driver = webdriver.Chrome(chrome_driver_binary, chrome_options=options)
base_url = r'https://android.myapp.com/myapp/detail.htm?apkName='
def download(app_name):
    #options = webdriver.ChromeOptions()
    #options.add_argument('-headless')
    #browser = webdriver.Chrome(chrome_options=options,executable_path='chromedriver')
    
    #google:
    #'https://play.google.com/store/apps/details?id=com.gravity.romg'
    head={'User-Agent':'Mozilla/5.0'}
    s=requests.Session()
    scont=s.get('https://android.myapp.com/myapp/detail.htm?apkName='+app_name,headers=head)
    html = etree.HTML(scont.text)
    html_data =html.xpath('//div[@class="det-app-data-info"]/text()')
    x=[0]
    x='。'.join(html_data)
    x=x.replace('\r','')
    if len(x)<2:
        return 0
    else:
        return x

    #browser.close()

if __name__ == "__main__":
    train= pd.read_csv("final_apptype_train.dat",header=None,encoding='utf8',delimiter='\t')
    train.columns=['id','conment','label']

    #test=pd.read_csv("appname_package.dat",header=None,encoding='utf8',delimiter='\t')
    feats=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        train_files=list(train.conment)
        for path,res in tqdm(zip(train_files,executor.map(download,train_files))):
            feats.append(res)
    
    #train['conment1']=feats
    #train[['conment1','label']].to_csv('new_train.csv',encoding='utf8')
    #print('未爬到app的个数：',train[train.conment1==0].shape[0])
    


