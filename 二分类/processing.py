# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:21:09 2018
@author: wushaowu
function:文件的转存
"""

import os
import PIL.Image as Image
import pandas as pd 
xx='new_NJ0130-0402_noPng/'
#path='旧的储存方式/'
#path='NJ0130-0402_noPng/' #待处理文件路径
files=os.listdir('789.png') #返回path下的所有文件名
ed
for i in files: 
    sub_files=os.listdir(path+i) #返回path+i下的所有文件名,(包括文件夹和一个csv文件)
    for j in sub_files[:-1]: #遍历path+i下的所有文件夹（不包括csv文件）
        fromImage = Image.open(path+i+'/'+j+'/'+j+'.png') #读取文件夹里的图片（文件名与文件夹名一样）
        f=open(path+i+'/'+j+'/'+j+'.csv')
        fromcsv = pd.read_csv(f) #读取文件夹里的csv文件（文件名与文件夹名一样）
        
        """在当前路径下建立i/png和i/csv文件夹"""
        isexist_png= os.path.exists(xx+i+'/'+'png') #是否存在i/png文件夹
        isexist_png= os.path.exists(xx+i+'/'+'csv') #是否存在i/csv文件夹
        if not isexist_png:
            os.makedirs(xx+i+'/'+'png') #建立i/png文件夹
            os.makedirs(xx+i+'/'+'csv')
        else:
            pass
        
        fromImage.save(xx+i+'/'+'png'+'/'+j+'.png') #保存图片到新的文件夹里
        fromcsv.to_csv(xx+i+'/'+'csv'+'/'+j+'.csv',index=None) #保存csv文件到新的文件夹里
    f=open(path+i+'/'+sub_files[-1]) 
    user_infos = pd.read_csv(f) #读取 path+i下的csv文件，并保存到i文件夹下
    user_infos.to_csv(xx+i+'/'+sub_files[-1],index=None)
