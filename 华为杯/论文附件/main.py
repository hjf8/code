# -*- coding: utf-8 -*-
from utils import _detecting,_create_feature,pca,plotBestFit #导入函数
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = ['sans-serif']
#plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 100
import time
from numpy import *
if __name__ == "__main__":
    print('=====================__name__ start=====================')
    print('\n----->'+'读入初始数据...')
    t0 = time.time()
    file_path='原始数据/'
    vehicle_1= pd.read_excel(file_path+'文件1.xlsx')
    vehicle_2= pd.read_excel(file_path+'文件2.xlsx')
    vehicle_3= pd.read_excel(file_path+'文件3.xlsx')
    vehicle_list=[vehicle_1,vehicle_2,vehicle_3]
    
    fragment=[] #保存运动学片段数
    new_data=pd.DataFrame() #保存预处理后的运动学片段
    new_feats=pd.DataFrame() #保存运动学片段特征
    for i in range(len(vehicle_list)):
        print('\n----->'+'第%d辆车检测中...\n'%(i+1))
        data,initial_frag,del_frag0,del_frag1,del_frag2,del_frag3,final_frag=_detecting(vehicle_list[i])
        data['车辆编号']=i+1
        feats=_create_feature(data,i+1)
        fragment.append([i+1,initial_frag,del_frag0,del_frag1,del_frag2,del_frag3,final_frag])
        new_data=pd.concat([new_data,data],axis=0)
        new_feats=pd.concat([new_feats,feats],axis=0)
    fragment=pd.DataFrame(fragment,columns=['车辆编号','初始运动学片段数',\
                                            '采集信号异常片段数',\
                                            '最大速度小于10km/h的片段数',\
                                            '处理为后一片段怠速部分的片段数',\
                                            '加速度/怠速异常片段数','最终有效片段数'])

    print('\n----->'+'去除异常片段(根据基本假设去除)...')
    new_feats=new_feats[(new_feats['匀速时间(s)']>=0)&(new_feats['行驶距离(m)']<50000)&\
                        (new_feats['平均速度(m/s)']<34)&(new_feats['平均行驶速度(m/s)']<34)].reset_index(drop=True)
    
    fragment.to_csv('三辆车运动学片段数检测结果表.csv',index=None,encoding='gbk')
    new_data.to_csv('三辆车预处理后的运动学片段数据表.csv',index=None,encoding='gbk')
    new_feats.to_csv('三辆车运动学片段特征参数表.csv',index=None,encoding='gbk')
    print('\n----->'+'运动学片段提取完毕！')
    print('\n----->'+'运动学片段特征提取完毕！')