# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:27:49 2019

@author: shaowu
"""

import gc
import os
import re
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import xgboost as xgb
#import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn import preprocessing
from tqdm import *
import time
from ahp import AHP
seed = 42
np.random.seed = seed

path='附件1-示例数据-100辆车/'
data_list = os.listdir(path)
from folium import plugins
import folium
def draw_routes(df):
    '''
    function:
        画经纬度轨迹图；
    Parameters:
        df --DataFrame类型数据表;
        return --该经纬度序列的轨迹图；
    '''
    m=folium.Map([df['lat'].mean(),\
              df['lng'].mean()],zoom_start=6)

    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    ##求每次驾驶的起终始经纬度，并在地图上标记：
    if df['cishu'].max()<1:
        print('此车无路程！')
    for i in range(1,df['cishu'].max()+1):
        start_lat=df[df.cishu==i].iloc[0]['lat'] #起始纬度
        start_lng=df[df.cishu==i].iloc[0]['lng'] #起始经度
        end_lat=df[df.cishu==i].iloc[-1]['lat'] #结束纬度
        end_lng=df[df.cishu==i].iloc[-1]['lng'] #结束经度
        ##画箭头：使用结束前的一个经纬度，和结束时的经纬度:
        k=2
        while k<df[df.cishu==i].shape[0]:
            if df[df.cishu==i].iloc[-1]['lat']!=df[df.cishu==i].iloc[-k]['lat'] or \
        df[df.cishu==i].iloc[-1]['lng']!=df[df.cishu==i].iloc[-k]['lng']:
                sencod_end_lat=df[df.cishu==i].iloc[-k]['lat'] #结束纬度
                second_end_lng=df[df.cishu==i].iloc[-k]['lng'] #结束经度
                break
            else:
                k=k+1
        '''
        ##箭头：
        jtou=folium.PolyLine([[sencod_end_lat,second_end_lng],[end_lat,end_lng]],\
                      weight=1,
                      color='red',\
                      opacity=0.8,\
                      control_scale=True,\
                      ).add_to(m)
        '''
        #画起始点：
        folium.Marker([start_lat, start_lng],\
                   icon=folium.Icon(color='blue',\
                                    icon='起始'),
                   popup="{0}:{1}".format('起始', [start_lat, start_lng])
                   ).add_to(marker_cluster)
        #folium.Tooltip(text='起始',style = None,sticky = True)
        #画终止点“
        folium.Marker([end_lat, end_lng],\
                   icon=folium.Icon(color='red',\
                                    icon='结束'),
                   popup="{0}:{1}".format('结束', [end_lat, end_lng])
                   ).add_to(marker_cluster)
        #folium.Tooltip(text='结束',style = None,sticky = True)
        #PolyLineTextPath(折线,文本,\
        #repeat = False,center = False,Below = False,offset = 0,orientation = 0,attributes = None )
        #Tooltip(text,style = None,sticky = True)
        
    location=[[row['lat'],row['lng']] for i,row in df.iterrows()] #经纬度list表

    route=folium.PolyLine(location,\
                      weight=2,
                      color='red',\
                      opacity=0.8,\
                      control_scale=True,\
                      ).add_to(m)
    mkdir('routes_C/') ##建立文件夹
    m.save(os.path.join('routes_C/','heatmap_%d.html'%time.time())) #保存为html
def mkdir(path):
    '''新建文件夹path'''
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        return True
    else:
        print('folder is exist')
        return False
def time_transform(x):
    '''
    Function:
        时间转换成时间戳，方便计算时间差;
    Parameters:
        x --时间格式为%Y-%m-%d %H:%M:%S;
    return 时间戳格式
    '''
    return time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S"))

#---------------所以文件结合，方便处理（电脑内存足够大可以这样做）-----------
if not os.path.exists('alldata_C.csv'):
    df = pd.read_csv(path+ data_list[0])
    df.to_csv('alldata_C.csv', index=False)

    for i in tqdm(range(1, len(data_list))):
        if data_list[i].split('.')[-1] == 'csv':
            df = pd.read_csv(path + data_list[i])
            df.to_csv('alldata_C.csv', index=False, header=False, mode='a+')
        else:
            continue
 
#经纬度转换成距离       
from math import sin, asin, cos, radians, fabs, sqrt
 
EARTH_RADIUS=6371           # 地球平均半径，6371km
 
def hav(theta):
    s = sin(theta / 2)
    return s * s
 
def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)
 
    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
 
    return distance        
        
        
#alldata= pd.read_csv('alldata_C.csv') ##内存足够大的情况
#----------------对每一个文件，遍历处理（方便内存小的）--------------------
Z=[]
R=[]
for i in tqdm(data_list[20:]): #遍历每一个文件
    sub_z=[]
    sub_R=[]
    df=pd.read_csv(path+i) #读取文件
    cishu=[] #用于存放驾驶次数，从1开始算起，0代表无驾驶
    shijian=[] #用于存放每次的点火时间到熄火时间
    k=1
    for j in range(len(df)):
        if df.loc[j,'acc_state']==1: #点火状态
            #cishu.append(k)
            shijian.append(j)###################################
                
        elif len(shijian)>0: #此时刻为熄火状态，判断熄火前gps的速度
            if df.loc[shijian]['gps_speed'].max()>0: #熄火前的gps速度是否存在大于0
                cishu=cishu+[k]*len(shijian) #此刻熄火前的驾驶次数
                cishu.append(0) #此处为当前时刻熄火时的情况，补0
                k=k+1 # 准备下一次驾驶次数
                shijian=[] #清空shijian
            else: #熄火前gps没有速度，不算一次驾驶次数
                cishu=cishu+[0]*len(shijian) #无驾驶情况，补0
                cishu.append(0) #此处为当前时刻熄火时的情况，补0
                shijian=[] #清空shijian
        else: #无驾驶情况，直接补0
            cishu.append(0)
        if (j==len(df)-1) & (len(shijian)>0): #防止最后一次都是点火的情况，或全为点火运输的情况
            cishu=cishu+[k]*len(shijian)
    df['cishu']=cishu #驾驶次数，从1开始算起
    df['time_stamp']=df['location_time'].apply(lambda x:time_transform(x)) #时间转换成时间戳
    df['day']=df['location_time'].apply(lambda x:x[:10]) #提取日期：年月日
    df['week']=df['location_time'].apply(lambda x:pd.to_datetime(x).weekday()+1) #提取星期
    draw_routes(df)
    #超速
    df['chaosu']=df['gps_speed'].apply(lambda x:1 if x>60 else 0)
    m=[]
    n=[]
    h=[]#每次超速时间
    for i in range(len(df)-3):
        if df['chaosu'][i+1]-df['chaosu'][i]==1:
            m.append(i+1)#起始
        elif df['chaosu'][i+1]-df['chaosu'][i]==-1:
            n.append(i+1)#结束
        if df['chaosu'][i] == 1 and df['chaosu'][i+1]==1:
            if df['mileage'][i+1]-df['mileage'][i]>=2:
                n.append(i)
                m.append(i+1)
            if df['time_stamp'][i+1]-df['time_stamp'][i]>=10:
                n.append(i)
                m.append(i+1)
    m.sort()
    n.sort()
   
    if len(m)>len(n):
        #if len(df)-m[len(m)-1]>=2:
        #    n=list(n)+list([len(df)-1])
       # else:
        m=m[:-1]
         #n=list(n)+list([len(df)-1])
        #m=m[:-1]
    for i in range(len(m)):
        h.append(df['time_stamp'][n[i]]-df['time_stamp'][m[i]])
    cishu_chao=len(h)
    
    if len(h)!=0:
        average_chaoshu_time=sum(h)/(len(h))/60#平均每次超速的时间(分)
    else:
        average_chaoshu_time = 0
            
    m=[]
    n=[]
    h=[]
    #急加,急减
    df['gps']=list(df['gps_speed'][1:])+list([-1])
    df['加速度']=(df['gps']-df['gps_speed'])*1000/3600/1
    df.drop(['gps'],axis=1,inplace=True)
    #急加
    df['加']=df['加速度'].apply(lambda x:1 if x>5 else 0)
    for i in range(len(df)-1):
        if df['加'][i+1]-df['加'][i]==1:
            m.append(i+1)#起始
        elif df['加'][i+1]-df['加'][i]==-1:
            n.append(i+1)#结束
        #if df['加'][i] == 1 and df['加'][i+1]==1:
         #   if df['mileage'][i+1]-df['mileage'][i]>=2:
         #       n.append(i)
          #      m.append(i+1)
          #  if df['time_stamp'][i+1]-df['time_stamp'][i]>=10:
           #     n.append(i)
          #      m.append(i+1)
    #m.sort()
    #n.sort()
    if len(m)>len(n):
        #if len(df)-m[len(m)-1]>=2:
        #    n=list(n)+list([len(df)-1])
        #else:
        m=m[:-1]
         #n=list(n)+list([len(df)-1])
    cishu_jia = len(m)
    chao_jia = 0#急加速超过5的部分
    for i in range(len(m)):
        chao_jia =chao_jia + (df['加速度'][m[i]:n[i]].max()-5)
    if len(m)!=0:
        average_jijia = chao_jia/cishu_jia#平均急加超过5的部分  
    df.drop(['加'],axis=1,inplace=True)
    
    #急减
    m=[]
    n=[]
    df['减']=df['加速度'].apply(lambda x:1 if x<-5 else 0)
    for i in range(len(df)-1):
        if df['减'][i+1]-df['减'][i]==1:
            m.append(i+1)#起始
        elif df['减'][i+1]-df['减'][i]==-1:
            n.append(i+1)#结束
        if df['减'][i] == 1 and df['减'][i+1]==1:
            if df['mileage'][i+1]-df['mileage'][i]>=2:
                n.append(i)
                m.append(i+1)
            if df['time_stamp'][i+1]-df['time_stamp'][i]>=10:
                n.append(i)
                m.append(i+1)
    m.sort()
    n.sort()
    if len(m)>len(n):
        #if len(df)-m[len(m)-1]>=2:
        #    n=list(n)+list([len(df)-1])
        #else:
        m=m[:-1]
    cishu_jian = len(m)
    chao_jian = 0#急减速超过5的部分
    for i in range(len(m)):
        chao_jian =chao_jian + abs(df['加速度'][m[i]:n[i]].max()+5)
    if cishu_jian != 0:
        average_jijian = chao_jian/cishu_jian#m/s2
    df.drop(['减'],axis=1,inplace=True)
    
    m=[]#清空
    n=[]
    h=[]
    #疲劳驾驶
    df['驾驶时间']=list([0])+list(df['acc_state'][:-1])
    for i in range(len(df)-1):
        if df['驾驶时间'][i+1]-df['驾驶时间'][i]==1: #and (df['lng'][i+1]!=df['lng'][i] or df['lat'][i+1]!=df['lat'][i])):
            m.append(i)#起始   最上面加了一行0，所以减掉一行
        elif df['驾驶时间'][i+1]-df['驾驶时间'][i]==-1:# and (df['lng'][i+1]!=df['lng'][i] or df['lat'][i+1]!=df['lat'][i])):
            n.append(i-1)#结束
        if df['驾驶时间'][i] == 1 and df['驾驶时间'][i+1]==1:
            if df['mileage'][i+1]-df['mileage'][i]>=2:
                n.append(i)
                m.append(i+1)
            if df['time_stamp'][i+1]-df['time_stamp'][i]>=10:
                n.append(i)
                m.append(i+1)
    m.sort()
    n.sort()
    if len(m)>len(n):
        #if len(df)-m[len(m)-1]>=2:
        #    n=list(n)+list([len(df)-1])
        #else:
        m=m[:-1]
        #n=list(n)+list([len(df)-1])
        #m=m[:-1]
    
    
    m1=[]#位置发生变化 
    m2=[]
    #for i in range(len(m)):
    #    if df['lng'][n[i]]-df['lng'][int((m[i]+n[i])/2)]!=0 or df['lat'][n[i]]-df['lat'][int((m[i]+n[i])/2)]!=0:
    #        m1.append(i)
    for i in range(len(m)):
        for j in range(len(df['lng'][m[i]:n[i]])-1):
            if df['lng'][m[i]:n[i]].iloc[j+1]-df['lng'][m[i]:n[i]].iloc[j]!=0 or df['lat'][m[i]:n[i]].iloc[j+1]-df['lat'][m[i]:n[i]].iloc[j]!=0:
                m2.append(m[i]+j)
                m1.append(i)
                break 
    #驾驶时间(动)
    for i in range(len(m1)):
        h.append((df['time_stamp'][n[m1[i]]]-df['time_stamp'][m[m1[i]]])/3600)
    p0=0
    p1=0
    p2=0
    p3=0
    p4=0
    h_pilao=0
    for i in range(len(h)):
        if h[i]>4:
            p0=p0+1#疲劳驾驶次数
            if h[i]<=6:
                p1=p1+1#疲劳驾驶超过2小时以内
            elif 6<h[i]<=8:
                p2=p2+1#2-4小时
            elif 8<h[i]<=10:
                p3=p3+1#4-6
            elif h[i]>10:
                p4=p4+1#6以上
            h_pilao = h_pilao+(h[i]-4)
    if h_pilao>0:
        average_pilao = h_pilao/p0
    else:
        average_pilao = 0
        
    
    '''
    for i in range(len(m1)):
        for j in range(len(df['lng'][m[m1[i]]:n[m1[i]]])-1):
            if df['lng'][m[m1[i]]:n[m1[i]]].iloc[j+1]-df['lng'][m[m1[i]]:n[m1[i]]].iloc[j]!=0 or df['lat'][m[m1[i]]:n[m1[i]]].iloc[j+1]-df['lat'][m[m1[i]]:n[m1[i]]].iloc[j]!=0:
                m2.append(m[i]+j)
                m3.append(i)
                break
    '''
    #怠速预热
    h1=[]
    #一直没动    
    for i in range(len(m)):
        if i not in m1:
            h1.append((df['time_stamp'][n[i]]-df['time_stamp'][m[i]])/3600)      
    #动了
    for i in range(len(m1)):
        h1.append((df['time_stamp'][m2[i]]-df['time_stamp'][m[m1[i]]])/3600)
    average_daisu = np.mean(h1)*60    #分
    std_daisu=np.std(h1,ddof=1)
         
   
    #熄火滑行
    x1=[]
    x2=0
    for i in range(len(df)-1):
        if df['acc_state'][i] == 0 and df['acc_state'][i+1] == 0:
            if 0<abs(df['lng'][i+1]-df['lng'][i])<0.001 and 0<abs(df['lat'][i+1]-df['lat'][i])<0.001:
                lon1,lat1 = (df['lng'][i], df['lat'][i]) #深圳野生动物园(起点）
                lon2,lat2 = (df['lng'][i+1], df['lat'][i+1]) #深圳坪山站 (百度地图测距：38.3km)
                x2 =x2 + get_distance_hav(lon1,lat1,lon2,lat2)
                x1.append(i)
    if x2!=0:
        m_xihuo = x2/len(x1)
    else:
        m_xihuo=0
            #x2=x2+(df['mileage'][i+1]-df['mileage'][i])
           
    sub_z=sub_z+[p0,average_pilao,average_daisu,average_jijia,cishu_jia,average_jijian,cishu_jian,average_chaoshu_time,cishu_chao,m_xihuo]
    Z.append(sub_z)
    sub_R = AHP(p0,average_pilao,average_chaoshu_time,cishu_chao,average_jijian,cishu_jian,average_jijia,cishu_jia,m_xihuo,average_daisu)
    #sub_R.append(R)
    R.append(sub_R)
    
    
 
    
    
    
    
    
    '''
    z=df[df.chishu==1]
    ed
    m=1
    n=0
    chaoshu=[]#超速的次数
    shijian_chaosu=[]#超速的持续时间
    for j in range(len(df)):
        if df.loc[j,'gps_speed']>60:
            chaosu.append(m)
        else:
            chaosu.append(n)
    df['chaoshu']=chaoshu
    ed        
    '''
    '''
    挖掘分析：
    分析一次运输完成的过程：
    点火到起步：怠速预热时间，是否打转向灯
    起步到熄火：总时间；总路程；平均速度；最大速度；最后是否打转向灯；最后是否拉手刹后熄火；每次脚刹时长（相当于减速情况）；
    每次加速时长；变道次数；超车次数；
    行驶中，打开左转向灯时到方向角改变（变大或变小）时的时长（相当于变道情况）；
    方向角改变时，是否加速（未加速可为变道情况，加速可为超车情况）；若加速，加速过程中，是否出现打右转向灯，变道回来，成功
    算一次超车，否则算变道；
    总体分析：
    结合每一次运输完成的各项指标，求其指标的统计量（mean,min,max,count,sum等）；
    休息时间间隔的统计量,每天运输平均次数，每天运输平均时间，每天运输平均距离;
    每周运输平均次数，每周运输平均时间，每周运输平均距离；
    习惯运输时间段，
    '''
    '''
    #=====================================================================
    #分析每一次运输
    if df['cishu'].max<1:
        print('此车未有运输！')
    else:
        for chishu in range(df['cishu'].max()):
            print('每一次运输异常检测...')
            z=df[df.chishu==chishu]
            ed
    #====================================================================
    
    ed
    '''
'''
#------------------------------------------------------------------------------
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
def make_map(scale,box):
    fig=plt.figure(figsize=(8, 10))
    ax=plt.axes(projection=ccrs.PlateCarree())
    #set_extent需要配置相应的crs，否则出来的地图范围不准确
    ax.set_extent(box,crs=ccrs.PlateCarree())
    land = cfeature.NaturalEarthFeature('physical', 'land', scale,edgecolor='face',
                                                              facecolor=cfeature.COLORS['land'])
    ax.add_feature(land, facecolor='0.75')
    ax.coastlines(scale)
    ax.stock_img()
    #标注坐标轴
    #ax.set_xticks(np.arange(box[0],box[1]+xstep,xstep), crs=ccrs.PlateCarree())
    #ax.set_yticks(np.arange(box[2],box[3]+ystep,ystep), crs=ccrs.PlateCarree())
    ax.set_xticks([box[0],box[1]], crs=ccrs.PlateCarree())
    ax.set_yticks([box[2],box[3]], crs=ccrs.PlateCarree())    
    #zero_direction_label用来设置经度的0度加不加E和W
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #添加网格线
    ax.grid()
    return fig,ax
box=[df['lng'].min(),df['lng'].max(),df['lat'].min(),df['lat'].max()]
fig,ax=make_map(scale='10m',box=box)
#创建与labels对应的经纬度标注位置
xticks=df['lng']
yticks=df['lat']
labels=[i for i in range(len(df[['lng','lat']].drop_duplicates()))]
#标注经纬度    
#ax.text(0.01,0.23,'60$^\circ$W',transform=ax.transAxes,rotation=25)
#ax.text(-63,50,'60$^\circ$W',transform=ccrs.Geodetic(),rotation=25)

for xtick,ytick,label in zip(xticks,yticks,labels):
    ax.text(xtick,ytick,label,transform=ccrs.Geodetic(),size=2,color='r')
'''
#-------------------------------------------------------------------------------
from folium import plugins
import folium
def draw_routes(df):
    '''
    function:
        画经纬度轨迹图；
    Parameters:
        df --DataFrame类型数据表;
        return --该经纬度序列的轨迹图；
    '''
    m=folium.Map([df['lat'].mean(),\
              df['lng'].mean()],zoom_start=6)

    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    ##求每次驾驶的起终始经纬度，并在地图上标记：
    if df['cishu'].max()<1:
        print('此车无路程！')
    for i in range(1,df['cishu'].max()+1):
        start_lat=df[df.cishu==i].iloc[0]['lat'] #起始纬度
        start_lng=df[df.cishu==i].iloc[0]['lng'] #起始经度
        end_lat=df[df.cishu==i].iloc[-1]['lat'] #结束纬度
        end_lng=df[df.cishu==i].iloc[-1]['lng'] #结束经度
        ##画箭头：使用结束前的一个经纬度，和结束时的经纬度:
        k=2
        while k<df[df.cishu==i].shape[0]:
            if df[df.cishu==i].iloc[-1]['lat']!=df[df.cishu==i].iloc[-k]['lat'] or \
        df[df.cishu==i].iloc[-1]['lng']!=df[df.cishu==i].iloc[-k]['lng']:
                sencod_end_lat=df[df.cishu==i].iloc[-k]['lat'] #结束纬度
                second_end_lng=df[df.cishu==i].iloc[-k]['lng'] #结束经度
                break
            else:
                k=k+1
        '''
        ##箭头：
        jtou=folium.PolyLine([[sencod_end_lat,second_end_lng],[end_lat,end_lng]],\
                      weight=1,
                      color='red',\
                      opacity=0.8,\
                      control_scale=True,\
                      ).add_to(m)
        '''
        #画起始点：
        folium.Marker([start_lat, start_lng],\
                   icon=folium.Icon(color='blue',\
                                    icon='起始'),
                   popup="{0}:{1}".format('起始', [start_lat, start_lng])
                   ).add_to(marker_cluster)
        #folium.Tooltip(text='起始',style = None,sticky = True)
        #画终止点“
        folium.Marker([end_lat, end_lng],\
                   icon=folium.Icon(color='red',\
                                    icon='结束'),
                   popup="{0}:{1}".format('结束', [end_lat, end_lng])
                   ).add_to(marker_cluster)
        #folium.Tooltip(text='结束',style = None,sticky = True)
        #PolyLineTextPath(折线,文本,\
        #repeat = False,center = False,Below = False,offset = 0,orientation = 0,attributes = None )
        #Tooltip(text,style = None,sticky = True)
        
    location=[[row['lat'],row['lng']] for i,row in df.iterrows()] #经纬度list表

    route=folium.PolyLine(location,\
                      weight=2,
                      color='red',\
                      opacity=0.8,\
                      control_scale=True,\
                      ).add_to(m)
    mkdir('routes_C/') ##建立文件夹
    m.save(os.path.join('routes_C/','heatmap_%d.html'%time.time())) #保存为html
