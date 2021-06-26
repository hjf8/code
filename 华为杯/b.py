# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:39:19 2020

@author: Sun
"""

import xlrd
import pandas as pd
file = '附件三：285号和313号样本原始数据.xlsx'
def excel_to_list(data_file, sheet):
    data_list = []  # 新建个空列表，来乘装所有的数据
    wb = xlrd.open_workbook(data_file)  # 打开excel
    sh = wb.sheet_by_name(sheet)  # 获取工作簿
    for i in range(sh.nrows):  # 跳过标题行，从第二行开始取数据
        d = sh.row_values(i)
        data_list.append(d)
        #d = dict(zip(header, sh.row_values(i)))  # 将标题和每行数据组装成字典
        #data_list.append(d)
    data_list = pd.DataFrame(data_list) 
    return data_list  # 列表嵌套字典格式，每个元素是一个字典
yuanliao = excel_to_list(file, '原料') 
chanpin = excel_to_list(file, '产品') 
xifuji_daisheng = excel_to_list(file, '待生吸附剂') 
xifuji_zaisheng = excel_to_list(file, '再生吸附剂') 
var_caozuo = excel_to_list(file, '操作变量') 
#简单处理
chanpin = (chanpin.drop(index=[1])).reset_index(drop = True)
xifuji_daisheng = (xifuji_daisheng.drop(index=[1])).reset_index(drop = True)
xifuji_zaisheng = (xifuji_zaisheng.drop(index=[1])).reset_index(drop = True)
#col = var_caozuo.iloc[2]
#col[0] = '时间'
var_caozuo_285 = var_caozuo[3:43]
var_caozuo_285 = var_caozuo_285.reset_index(drop = True)
#var_caozuo_285.columns = col
var_caozuo_313 = var_caozuo[44:]
var_caozuo_313 = var_caozuo_313.reset_index(drop = True)
#var_caozuo_313.columns = col

#对于只含有部分时间点的位点，如果其残缺数据较多，无法补充，将此类位点删除；
#首先去掉里面的空格
temp_285 = var_caozuo_285.isnull().any() 
print('285号样本是否存在缺失值', True in list(temp_285))#经检测，285样本无缺失值
temp_313 = var_caozuo_313.isnull().any() 
print('313号样本是否存在缺失值', True in list(temp_313))#经检测，313样本无缺失值
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    