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
    t0 = time.time()
    new_feats=pd.read_csv('三辆车运动学片段特征参数表.csv')
    new_data=pd.read_csv('三辆车预处理后的运动学片段数据表.csv.csv')
    #================================分割线=============================================================
    print('\n----->'+'对运动学片段pca降维...')
    x,y,z=pca(np.matrix(new_feats.drop(['车辆编号','运动学片段',\
                                        '平均发动机转速','平均扭矩百分比','平均发动机负荷百分比'],axis=1)), 4)
    plotBestFit(x, y)
    
    print('\n----->'+'对运动学片段聚类...')
    from sklearn.cluster import KMeans
    clf=KMeans(n_clusters=3).fit(x)
    
    zz=clf.labels_
    print(pd.Series(zz).value_counts())
    
    x=np.array(x)
    markers = ['o', '*', '+']
    members = clf.labels_ == 0
    m1=0;m2=1
    plt.scatter(x[members, m1], x[members, m2], s=60, marker=markers[0], c='b', alpha=0.5)
    members = clf.labels_ == 1
    plt.scatter(x[members, m1], x[members, m2], s=60, marker=markers[1], c='r', alpha=0.5)
    members = clf.labels_ == 2
    plt.scatter(x[members, m1], x[members, m2], s=60, marker=markers[2], c='g', alpha=0.5) 
    members = clf.labels_ == 3
    plt.title(' ')
    plt.savefig('topics/聚类结果示意图.png',dpi=100)
    plt.show()
    
    
    new_feats['cluster']=zz
    new_feats['time']=new_feats['加速时间(s)']+new_feats['减速时间(s)']+new_feats['怠速时间(s)']+new_feats['匀速时间(s)']
    time_all=new_feats.groupby(['cluster'],as_index=False)['time'].agg({'cluster_speed_sum':'sum'})
    
    
    print('\n----->'+'计算各类的运动学片段特征...')
    col='平均速度(m/s)'
    m=new_feats.groupby(['cluster'],as_index=False)[col].agg({'cluster_speed_mean':'mean'})
    m['cluster_speed_mean']=m['cluster_speed_mean']*3.6
    m_jia=new_feats.groupby(['cluster'],as_index=False)['加速时间(s)'].agg({'cluster_speed_sum':'sum'})
    bili_jia=m_jia['cluster_speed_sum']/time_all['cluster_speed_sum']
    
    m_jian=new_feats.groupby(['cluster'],as_index=False)['减速时间(s)'].agg({'cluster_speed_sum':'sum'})
    bili_jian=m_jian['cluster_speed_sum']/time_all['cluster_speed_sum']
    
    m_yun=new_feats.groupby(['cluster'],as_index=False)['匀速时间(s)'].agg({'cluster_speed_sum':'sum'})
    bili_yun=m_yun['cluster_speed_sum']/time_all['cluster_speed_sum']
    m_dai=new_feats.groupby(['cluster'],as_index=False)['怠速时间(s)'].agg({'cluster_speed_sum':'sum'})
    bili_dai=m_dai['cluster_speed_sum']/time_all['cluster_speed_sum']
    #表5
    
    
    v_ave=new_feats.groupby(['cluster'],as_index=False)['平均行驶速度(m/s)'].agg({'cluster_speed_mean':'mean'})#平均行驶速度
    v_ave['cluster_speed_mean']=v_ave['cluster_speed_mean']*3.6
    v_max=new_feats.groupby(['cluster'],as_index=False)['最大速度(m/s)'].agg({'cluster_speed_mean':'mean'})
    v_max['cluster_speed_mean']=v_max['cluster_speed_mean']*3.6
    a_max=new_feats.groupby(['cluster'],as_index=False)['最大加速度(m/s^2)'].agg({'cluster_speed_mean':'mean'})
    a_ave=new_feats.groupby(['cluster'],as_index=False)['平均加速度(m/s^2)'].agg({'cluster_speed_mean':'mean'})
    a_max_jian=new_feats.groupby(['cluster'],as_index=False)['最大减速度(m/s^2)'].agg({'cluster_speed_mean':'mean'})
    a_ave_jian=new_feats.groupby(['cluster'],as_index=False)['平均减速度(m/s^2)'].agg({'cluster_speed_mean':'mean'})
    #标准差
    std_v=new_feats.groupby(['cluster'],as_index=False)['速度标准差'].agg({'cluster_speed_mean':'mean'})
    std_a=new_feats.groupby(['cluster'],as_index=False)['加速度标准差'].agg({'cluster_speed_mean':'mean'})
    #能耗，构建工况
    new_feats['能耗']=(new_feats['平均发动机负荷百分比']*new_feats['平均扭矩百分比']*new_feats['平均发动机负荷百分比']*(new_feats['平均发动机转速']/3600)/9550/(new_feats['平均速度(m/s)']*3.6*0.72))*0.1
    ###=========================================
    #第一类
    x1=new_feats[new_feats['cluster']==0][['运动学片段','车辆编号','time','平均速度(m/s)','平均行驶速度(m/s)','平均加速度(m/s^2)','平均减速度(m/s^2)','怠速时间比','加速时间比','减速时间比','匀速时间比','速度标准差','加速度标准差','能耗']].reset_index(drop=True)
    plt.hist(x1['能耗'],10)
    plt.savefig('topics/拥堵类能耗分布图.png',dpi=100)
    x1['e1'] = np.abs(x1.loc[:,['怠速时间比']]-bili_dai[0])
    x1['e2'] = np.abs(x1.loc[:,['加速时间比']]-bili_jia[0])
    x1['e3'] = np.abs(x1.loc[:,['减速时间比']]-bili_jian[0])
    x1['e4'] = np.abs(x1.loc[:,['匀速时间比']]-bili_yun[0])
    x1['e'] = (x1['e1']+x1['e2']+x1['e3']+x1['e4'])/4
    x1['C'] = x1.loc[:,['能耗']]/(np.max(x1['能耗'])-np.min(x1['能耗']))
    x1['E'] = x1['e']/x1['C']
    x1_sort=x1.sort_values(by="E" , ascending=True) 
    x11=x1_sort[:13]
    x11.rename(columns={'运动学片段':'fragment'}, inplace = True)
    x12=x11.merge(new_data,how='left',on=['车辆编号','fragment'])
    plt.plot(x12['GPS车速'])
    plt.savefig('topics/拥堵类行驶工况曲线.png',dpi=100)
    #第二类
    x2=new_feats[new_feats['cluster']==1][['运动学片段','车辆编号','time','平均速度(m/s)','平均行驶速度(m/s)','平均加速度(m/s^2)','平均减速度(m/s^2)','怠速时间比','加速时间比','减速时间比','匀速时间比','速度标准差','加速度标准差','能耗']].reset_index(drop=True)
    plt.hist(x2['能耗'],10)
    plt.savefig('topics/一般类能耗分布图.png',dpi=100)
    x2['e1']=np.abs(x2.loc[:,['怠速时间比']]-bili_dai[1])
    x2['e2']=np.abs(x2.loc[:,['加速时间比']]-bili_jia[1])
    x2['e3']=np.abs(x2.loc[:,['减速时间比']]-bili_jian[1])
    x2['e4']=np.abs(x2.loc[:,['匀速时间比']]-bili_yun[1])
    x2['e']=(x2['e1']+x2['e2']+x2['e3']+x2['e4'])/4
    x2['C'] = x2.loc[:,['能耗']]/(np.max(x2['能耗'])-np.min(x2['能耗']))
    x2['E'] = x2['e']/x2['C']
    x2_sort=x2.sort_values(by="E" , ascending=True) 
    x21=x2_sort[:3]
    x21.rename(columns={'运动学片段':'fragment'}, inplace = True)
    x22=x21.merge(new_data,how='left',on=['车辆编号','fragment'])
    plt.plot(x22['GPS车速'])
    plt.savefig('topics/一般类行驶工况曲线.png',dpi=100)
    #第三类
    x3=new_feats[new_feats['cluster']==2][['运动学片段','车辆编号','time','平均速度(m/s)','平均行驶速度(m/s)','平均加速度(m/s^2)','平均减速度(m/s^2)','怠速时间比','加速时间比','减速时间比','匀速时间比','速度标准差','加速度标准差','能耗']].reset_index(drop=True)
    plt.hist(x3['能耗'],10)
    plt.savefig('topics/通畅类能耗分布图.png',dpi=100)
    x3['e1']=np.abs(x3.loc[:,['怠速时间比']]-bili_dai[2])
    x3['e2']=np.abs(x3.loc[:,['加速时间比']]-bili_jia[2])
    x3['e3']=np.abs(x3.loc[:,['减速时间比']]-bili_jian[2])
    x3['e4']=np.abs(x3.loc[:,['匀速时间比']]-bili_yun[2])
    x3['e']=(x3['e1']+x3['e2']+x3['e3']+x3['e4'])/4
    x3['C'] = x3.loc[:,['能耗']]/(np.max(x3['能耗'])-np.min(x3['能耗']))
    x3['E'] = x3['e']/x3['C']
    x3_sort=x3.sort_values(by="E" , ascending=True) 
    x31 = x3_sort[:2]
    x31.rename(columns={'运动学片段':'fragment'}, inplace = True)
    x32=x31.merge(new_data,how='left',on=['车辆编号','fragment'])
    plt.plot(x32['GPS车速'])
    plt.savefig('topics/通畅类行驶工况曲线.png',dpi=100)
    #综合工况的构建
    x12_1 = x11[:6]
    x22_1 = x21[:1]
    x32_1 = x31[:1]
    x_pingjie=pd.concat([x12_1,x22_1,x32_1]).reset_index(drop=True)
    x_pingjie=x_pingjie.merge(new_data,how='left',on=['车辆编号','fragment'])
    plt.plot(x_pingjie['GPS车速'])
    plt.savefig('topics/轻型汽车行驶工况曲线.png',dpi=100)
    #整合
    
    #全部数据
    V_m_all = np.mean(new_feats['平均速度(m/s)'])*3.6  #平均速度
    V_x_all = np.mean(new_feats['平均行驶速度(m/s)'])*3.6  #平均行驶速度
    A_a_all = np.mean(new_feats['平均加速度(m/s^2)'])  #平均加速度
    A_d_all = np.mean(new_feats['平均减速度(m/s^2)'])  #平均减速度
    bi_dai_all = np.mean(new_feats['怠速时间比'])  #平均怠速时间比
    bi_jia_all = np.mean(new_feats['加速时间比'])  #平均加速时间比
    bi_jian_all = np.mean(new_feats['减速时间比'])  #平均减速时间比
    bi_yun_all = np.mean(new_feats['匀速时间比'])  #平均匀速时间比
    std_v_all = np.mean(new_feats['速度标准差'])  #平均速度标准差
    std_a_all = np.mean(new_feats['加速度标准差'])  #平均加速度标准差
    #工况
    V_m_gong = np.mean(x_pingjie['平均速度(m/s)'])*3.6  #平均速度
    V_x_gong = np.mean(x_pingjie['平均行驶速度(m/s)'])*3.6  #平均行驶速度
    A_a_gong = np.mean(x_pingjie['平均加速度(m/s^2)'])  #平均加速度
    A_d_gong = np.mean(x_pingjie['平均减速度(m/s^2)'])  #平均减速度
    bi_dai_gong = np.mean(x_pingjie['怠速时间比'])  #平均怠速时间比
    bi_jia_gong = np.mean(x_pingjie['加速时间比'])  #平均加速时间比
    bi_jian_gong = np.mean(x_pingjie['减速时间比'])  #平均减速时间比
    bi_yun_gong = np.mean(x_pingjie['匀速时间比'])  #平均匀速时间比
    std_v_gong = np.mean(x_pingjie['速度标准差'])  #平均速度标准差
    std_a_gong = np.mean(x_pingjie['加速度标准差'])  #平均加速度标准差