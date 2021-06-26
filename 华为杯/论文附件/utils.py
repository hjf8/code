# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = ['sans-serif']
#plt.rcParams['savefig.dpi'] = 200 #图片像素
plt.rcParams['figure.dpi'] = 100
from sklearn import preprocessing
import time
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

def _detecting(vehicle,canshu1=180,canshu2=180,canshu3=3.96,canshu4=-8):
    '''
    功能：检测运动学片段
    参数：
    canshu1 设置采集中断允许的最大时间限制（单位：s）,默认180
    canshu2 设置车辆断断续续低速行驶允许的最小时间限制（单位：s）,默认180
    canshu3 设置加速度允许的最大加速度，单位（m/s^2）,默认3.96
    canshu4 设置减速度允许的最大减速度，单位（m/s^2）,默认-8
    返回：
    预处理后的运动学片段，初始运动学片段数，采集信号异常片段数，最大GPS速度小于10km/h的运动学片段数，加速度异常的片段数，有效片段数
    '''
    data=vehicle.copy()
    data['时间戳']=data['时间'].apply(lambda x:int(time.mktime(time.strptime(x.split('.')[0],"%Y/%m/%d %H:%M:%S"))))
    data=data.sort_values(by='时间戳',ascending=True).reset_index(drop=True)

    data['时间戳_diff']=data['时间戳'].diff()
    #plt.plot(data['时间戳_diff'])
    #plt.xlabel('时间(单位:s)')
    #plt.ylabel('时间间隔(单位:s)')

    data['0<GPS车速<10']=data['GPS车速'].apply(lambda x:1 if x>0 and x<10 else 0)

    print('第一步：计算原始运动学片段(怠速+运动，两部分组成！)...')
    useful_index=[] #初始化储存含GPS速度的指标
    not_useful_index=[] #初始化储存无GPS速度的指标
    data['fragment']=0 #初始化运动学片段为0
    k=1 #初始化运动学片段次数
    for i in tqdm(range(data.shape[0]-1)):
        if data.loc[i,'发动机转速']>0 and data.loc[i+1,'发动机转速']>0:
            if data.loc[i,'GPS车速']==0 and len(useful_index)<1: #第一次没有速度
                not_useful_index.append(i)
            elif len(not_useful_index)>0: #开始有速度，且有速度之前，有怠速
                if data.loc[i,'GPS车速']>0:  #有速度的情况
                    useful_index.append(i)
                else: #第二次没有速度，说明已经完成一次运输过程
                    whole_one_index=not_useful_index+useful_index #得到完整运动学片段
                    useful_index=[] #初始化
                    not_useful_index=[] #初始化
                    not_useful_index.append(i) #初始化之后，储存此时刻的指标，此时刻是没有速度的！
                    data.loc[whole_one_index,'fragment']=k #得到第k次运动学片段
                    k=k+1 #更新k
            else: #数据第一行就有GPS速度的情况
                pass
        else:
            print('发动机转速为0！')
    data=data[data.fragment>0].reset_index(drop=True)
    initial_fragment=len(data.fragment.unique())
    print('原始运动学片段数为%d片段'%initial_fragment)
    #==============================================================================
    print('第二步：对原始运动学片段处理...')
    #每个运动学片段中，最大时间间隔（即时间戳_diff的最大值）：
    fragment_timestamp_max=data.groupby(['fragment'],as_index=False)['时间戳_diff'].agg({'时间戳_diff_max':'max'})
    #plt.plot(fragment_timestamp_max.sort_values(by='时间戳_diff_max').reset_index(drop=True)['时间戳_diff_max'])

    #运动学片段中，没有出现长期停车，或采集中断的片段：
    del_fragment0=len(fragment_timestamp_max[fragment_timestamp_max['时间戳_diff_max']>=canshu1]['fragment'].unique())
    fragment_useful=fragment_timestamp_max[fragment_timestamp_max['时间戳_diff_max']<canshu1]['fragment'].unique().tolist()
    data=data[data.fragment.isin(fragment_useful)].reset_index(drop=True)

    #每个运动学片段中的最大GPS车速：
    fragment_speed_max=data.groupby(['fragment'],as_index=False)['GPS车速'].agg({'GPS车速_max':'max'})
    #返回运动学片段中速度小于10km/h的片段：
    fragment_pre=fragment_speed_max[fragment_speed_max['GPS车速_max']<10]['fragment'].unique().tolist()

    print('对于速度小于10km/h的运动学片段，如果当前运动学片段的后一个运动学片段，其怠速时间大于3分钟，\
          那么直接扔掉当前运动学片段；否则预处理成其后一运动学片段的怠速部分...')
    del_fragment=[] #存储要删除的运动学片段
    hou_fragment=[] #存储为后一运动学片段的怠速部分
    for i in fragment_pre:
        if data[data.fragment==i+1].shape[0]>0: #当前片段的后一个运动学片段存在
            m=data[(data.fragment==i+1)&(data['GPS车速']==0)] #后一段运动学片段的怠速部分
            _time=m['时间戳'].iloc[-1]-m['时间戳'].iloc[0] #后一段运动学片段的怠速时间
            if _time>180: #怠速时间已经大于3分钟，删除当前运动学片段
                del_fragment.append(i)
            elif data[data.fragment==i+1]['时间戳'].iloc[0]-\
            data[data.fragment==i]['时间戳'].iloc[-1]<180: #当前片段和下一片段，时间间隔小于3分钟
                data.loc[data.fragment==i,'GPS车速']=0 #当前运动学片段GPS速度置0
                data.loc[data.fragment==i,'fragment']=i+1 #当前片段，更改为后一运动学片段
                hou_fragment.append(i)
            else: #当前片段和下一片段，时间间隔大于3分钟，直接扔掉当前片段
                del_fragment.append(i)
    
        else: #不存在后一片段，直接扔掉当前片段
            del_fragment.append(i)
    print('直接扔掉速度小于10km/h的片段数为%d片段'%len(del_fragment))
    print('预处理成其后一运动学片段的片段数为%d片段'%len(hou_fragment))
    data=data[~data.fragment.isin(del_fragment)].reset_index(drop=True)
    
    #===========================================================================
    print('处理片段怠速部分，对大于3分钟的怠速部分，截取其后3分钟部分，并对加减速度进行处理...')
    new_data=pd.DataFrame() #存放预处理后最终的数据
    del_fragment1=[] #储存加速度含有异常的片段
    for i in data.fragment.unique():
        m=data[(data.fragment==i)&(data['GPS车速']==0)]
        if m['时间戳'].iloc[-1]-m['时间戳'].iloc[0]<180: #怠速时间小于3分钟
            _temp=m['时间戳'].iloc[0]
        else: #怠速时间超过3分钟，截取后半段的3分钟即可
            _temp=m['时间戳'].iloc[-1]-180 #怠速最后一刻，减去180秒，即得后半段3分钟的怠速部分
        mm=data[(data.fragment==i)&(data['时间戳']>=_temp)] #处理怠速部分后的完整片段
    
        #对加减速度进行处理：如果存在异常，那么直接扔掉！
        mm=mm.reset_index(drop=True)
        mm['GPS车速_diff']=mm['GPS车速'].diff()
        mm['时间戳_diff']=mm['时间戳'].diff()
        mm['加速度']=mm['GPS车速_diff']/mm['时间戳_diff']*10**3/3600 #注意单位换算
        if mm['加速度'].max()>canshu3 or mm['加速度'].min()<canshu4:
            del_fragment1.append(i)
        else:
            new_data=pd.concat([new_data,mm],axis=0)

    print('运动学片段含有加速度异常的片段数为%d片段'%len(del_fragment1))

    #对运动学片段的次数，重新编号：
    new_data=new_data.reset_index(drop=True)
    fragment_dict=zip(list(set(new_data.fragment)),range(len(set(new_data.fragment))))
    fragment_dict=dict(fragment_dict)
    new_data['fragment']=new_data['fragment'].map(fragment_dict)
    print('最终有效的运动学片段数为%d片段'%len(new_data.fragment.unique()))
    return new_data,initial_fragment,del_fragment0,len(del_fragment),len(hou_fragment),len(del_fragment1),len(new_data.fragment.unique())

def dist(lng1,lat1,lng2,lat2):
    """
    定义两点距离函数：
    lng1,lat1   经纬度
    lng2,lat2   经纬度
    返回：
    距离，单位：（km）
    """
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    #dis=2*asin(sqrt(a))*6371#*1000
    dis=2*asin(sqrt(a))*6378.137#*1000
    return dis
def _create_feature(vehicle,vehicle_num):
    '''
    功能：提取运动学片段特征
    参数：
        vehicle_num  车辆编号，比如1，表示第1辆车
    '''
    data=vehicle.copy()
    #为了计算距离，对经纬度为0的进行填充：用上一个值填充即可：
    mm=data[(data['经度']==0)|(data['纬度']==0)]
    for i in mm.index:
        if data[data.fragment==data.loc[i,'fragment']].index[0]!=i: #次缺失不是第一个的情况
            data.loc[i,'经度']=data.loc[i-1,'经度']
            data.loc[i,'纬度']=data.loc[i-1,'纬度']
        else: #用后面的填充,或者影响不大，可以忽略
            pass
    feats=[]
    for i in data.fragment.unique():
        #print('*'*50+'\n时间特征计算...')
        m=data[(data.fragment==i)].reset_index(drop=True)
        #m['时间戳_diff']=m['时间戳'].diff().fillna(0)
        #加速时间：即加速度大于等于0.1m/s^2的总点数:
        ta=m[(m['加速度']>=0.1)].shape[0]
        #ta=m[(m['加速度']>=0.1)]['时间戳_diff'].sum()
        #减速度时间，即减速度小于等于-0.1m/s^2的总点数:
        td=m[(m['加速度']<=-0.1)].shape[0]
        #td=m[(m['加速度']<=-0.1)]['时间戳_diff'].sum()
        #怠速时间:
        #t1=m[(m['GPS车速']!=0)]['时间戳'].iloc[0]-\
        #   m[(m['GPS车速']==0)]['时间戳'].iloc[0]
        t1=m[(m['GPS车速']==0)].shape[0]
        #匀速时间：
        #tc=m['时间戳'].iloc[-1]-m['时间戳'].iloc[0]-ta-td-t1
        tc=m.shape[0]-ta-td-t1
        #行停比：
        txt=ta+td+tc/t1
        
        #print('*'*50+'\n距离特征计算...')
        m=m[~((data['经度']==0)|(data['纬度']==0))].reset_index(drop=True)
        if True:
            #由经纬度计算距离：
            d=[]
            for j in range(1,len(m)):
                d.append(dist(m.loc[j-1,'经度'],m.loc[j-1,'纬度'],\
                m.loc[j,'经度'],m.loc[j,'纬度']))
            d=np.sum(d)*1000 #单位：m
        else: #由车速和时间来计算距离，s=v*t
            d=np.sum(m['GPS车速']*1000/3600) #单位：m
        
        #print('*'*50+'\n速度特征计算...')
        #最大速度：
        speed_max=m['GPS车速'].max()*1000/3600 #单位m/s
        #平均速度：
        speed_mean=d/(ta+td+tc+t1) #单位m/s
        #平均行车速度，不包括怠速部分：
        speed_mean_r=d/(ta+td+tc)
        #速度标准差：
        speed_std=np.sqrt(np.sum([(i*1000/3600-speed_mean)**2 for i in m['GPS车速']])/(len(m)))
        
        #print('*'*50+'\n加速度特征计算...')
        #最大加速度：
        a_max=m['加速度'].max()
        #平均加速度：
        if m[m['加速度']>=0.1].shape[0]>0: #有加速度的情况
            a_mean=np.sum(m[m['加速度']>=0.1]['加速度'])/ta
        else: #无加速度的情况
            a_mean=0
        #最大减速度：
        a_min=m['加速度'].min()
        #平均减速度：
        if m[m['加速度']<=-0.1].shape[0]>0:
            a_mean_d=np.sum(m[m['加速度']<=-0.1]['加速度'])/td
        else:
            a_mean_d=0
        #加速度标准差：
        if a_mean>0:
            a_std=np.sqrt(np.sum([(i-a_mean)**2 for i in m[m['加速度']>=0.1]['加速度']])/(len(m[m['加速度']>=0.1])))
        else:
            a_std=0
        #print('*'*50+'\n比例特征计算...')
        #怠速时间比例：
        P1=t1/(ta+td+tc+t1)
        #加速时间比例：
        Pa=ta/(ta+td+tc+t1)
        #减速度时间比例：
        Pd=td/(ta+td+tc+t1)
        #匀速时间比例：
        Pc=tc/(ta+td+tc+t1)
        #其他特征的计算：
        temp_mean1=np.mean(data[(data.fragment==i)]['发动机转速'])
        temp_mean2=np.mean(data[(data.fragment==i)]['扭矩百分比'])
        temp_mean3=np.mean(data[(data.fragment==i)]['发动机负荷百分比'])
        
        
        feats.append([i,\
                      vehicle_num,\
                      ta,td,t1,tc,txt,\
                      d,\
                      speed_max,speed_mean,speed_mean_r,speed_std,\
                      a_max,a_mean,a_min,a_mean_d,a_std,\
                      P1,Pa,Pd,Pc,\
                      temp_mean1,temp_mean2,temp_mean3])
    feats=pd.DataFrame(feats,columns=['运动学片段','车辆编号',\
                                      '加速时间(s)','减速时间(s)','怠速时间(s)','匀速时间(s)','行停时间比例',\
                                      '行驶距离(m)','最大速度(m/s)','平均速度(m/s)','平均行驶速度(m/s)','速度标准差',\
                                      '最大加速度(m/s^2)','平均加速度(m/s^2)','最大减速度(m/s^2)','平均减速度(m/s^2)','加速度标准差',\
                                      '怠速时间比','加速时间比','减速时间比','匀速时间比',\
                                      '平均发动机转速','平均扭矩百分比','平均发动机负荷百分比'])
    return feats


def meanX(dataX):
    '''求按行的平均值'''
    return np.mean(dataX,axis=0)#axis=0表示依照列来求均值。假设输入list,则axis=1
def pca(XMat, k):
    '''主成分分析,进行降维
    参数：
    XMat:表示待降维的数据矩阵
    k:表示降维后的维数
    '''
    average = meanX(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print('k must lower than feature number')
        return
    else:
        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return finalData, reconData,featValue
def plotBestFit(data1, data2):    
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        axis_x2.append(dataArr2[i,0]) 
        axis_y2.append(dataArr2[i,1])                 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()  