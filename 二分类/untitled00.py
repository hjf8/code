# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:17:44 2019

@author: Junfeng Hu
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
#import openpyxl
import xlrd
'''
#读取excel
ExcelFile=xlrd.open_workbook('血压手环采样数据.xlsx')
print(ExcelFile.sheet_names())
sheet=ExcelFile.sheet_by_name('Sheet1')
print(sheet.name,sheet.nrows,sheet.ncols)
rows=sheet.row_values(2)
cols=sheet.col_values(1)
cols=sheet.col_values(3)
P=[]
E=[]
for i in range(2,len(cols)):
    #print(i)
    if 'P ' in cols[i]:
        P.append(int(cols[i][2:]))
    elif 'E ' in cols[i]:
        a=int(cols[i][2:])
        E.append(a)
    else: 
        pass

'''
'''
plt.figure(figsize=(50,20))
plt.subplot(211)
plt.title('num_5')
#P=P[:760]
x=np.arange(len(P))
PP=np.array(P)
plt.plot(x,PP)
'''
'''
train=[]
train=P
train=pd.DataFrame(train)
train.columns = ['脉搏']
'''

train = pd.read_csv('PMD20181122002FL00LJ.csv',usecols=['脉搏'], encoding = 'gb18030')
#train.drop(['用户ID','序号‘], axis=1, inplace=True)

#散点图
#plt.figure(figsize=(8,6))
#plt.scatter([i for i in range(train.shape[0])],train['脉搏'])
#plt.plot([i/200 for i in range(len(train))],train['脉搏'])
#plt.xlabel('index', fontsize=12)
#plt.ylabel('yield', fontsize=12)
#plt.show()

#a=np.diff(train['脉搏'],1)
train['脉搏2']=list(train['脉搏'][2:])+list([-1,-1])
train['差']=(train['脉搏2']-train['脉搏'])*100
#plt.plot(train['差'][:-2])
plt.subplot(211)
plt.plot(train['脉搏'])#原ppg

aa=train['差'].tolist()
c=[]#增长最快点
k=0
for i in range(0,len(aa)-165,165):
    c.append(np.argmax(aa[i:i+165])+i)
#plt.plot(c,[train['脉搏'][i] for i in c]) 
#寻找谷点
d=[]
for i in range(len(c)-1):
    d.append(np.argmin(train['脉搏'][c[i]:c[i+1]]))
#plt.plot(d,[train['脉搏'][i] for i in d])
#样条插值
#数据准备
y=np.float64(train['脉搏'])

Z =np.array([])
for i in range(len(d)-1):
    X=d[i:i+2]#定义数据点
    Y=[train['脉搏'][k] for k in d[i:i+2]]
    x=np.arange(d[i],d[i+1],1) #定义观测点
    #进行一次样条拟合
    ipo3=spi.splrep(X,Y,k=1) #源数据点导入，生成参数
    iy3=spi.splev(x,ipo3) #根据观测点和样条参数，生成插值
    #plt.plot(x,iy3)
    z=(y[int(d[i]):int(d[i+1]-1)]-iy3[0:int(d[i+1]-d[i]-1)])
    Z = np.append(Z,z)
    #print(z.shape)
    #print(y[int(d[i]):int(d[i+1])]-iy3[0:int(d[i+1]-d[i])])
    #print(iy3[0:int(d[i+1]-d[i])])   
M=np.array(Z).reshape(-1,1)
#print(M.shape)
#print(M)
plt.subplot(212)
plt.plot(M)
d=[]
d.append(M)
pd.DataFrame(M,columns=['脉搏']).to_csv('M.csv',index=None,encoding='gbk')
# 传入插值节点序列x、原函数f和导函数f_，生成分段埃尔米特插值函数
'''
#三次样条插值
def lh(x,xi,yi,y_i):
        result = 0.0
        # 利用k来定位x所处的区间
        k = 0
        while k < len(xi) and not (xi[k] <= x and xi[k+1] >= x):
            k += 1
        result = ((x - xi[k+1]) / (xi[k] - xi[k+1]))**2 * (1 + 2* (x-xi[k])/ (xi[k+1]-xi[k])) * yi[k] +\
            ((x-xi[k])/(xi[k+1]-xi[k]))**2 * (1 + 2 * (x-xi[k+1])/(xi[k] - xi[k+1]) )*yi[k+1] + \
            ((x-xi[k+1])/(xi[k]-xi[k+1]))**2 * (x-xi[k]) * y_i[k] + \
            ((x-xi[k])/(xi[k+1]-xi[k]))**2 *(x-xi[k+1]) * y_i[k+1]
        return result
S=[] 
T=[]   
for i in range(len(d)-1):
    xi=np.array(d[i:i+2])
    yi=np.array([train['脉搏'][k] for k in d[i:i+2]])
    y_i=np.array([0,0])
    N=np.arange(d[i],d[i+1],1)
    for n in range(len(N)):
        x=N[n]
        s=lh(x,xi,yi,y_i)
        S = np.append(S,s)        
    t=(y[int(d[i]):int(d[i+1]-1)]-S[0:int(d[i+1]-d[i]-1)])
    T = np.append(T,t)
#ax=plt.plot(T)    
'''


