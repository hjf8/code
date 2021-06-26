
Created on Wed Apr 11 14:17:52 2018

@author: shaowu

import degenerate_kernel as dk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
if  name__ == '__main__':
    start_time=time.time()
    print('******************************程序入口*******************************')
    %lamda=int(input('pleas input lambda:'))
    nn=int(input('please input degree n:'))
    %a=int(input('please input the left value of interval:'))
    %b=int(input('please input the right value of interval:'))
    %m=int(input('please input the node of Gauss_Legendre:'))
    %print('计算中...')
    lamda=5;a=0;b=1;m=500
    error=[]
    for n in range(1,nn+1,1):%范围从1到nn+1,后面的1为步长，可以省略
        x,w=dk.gauss_xw(m) %取400个高斯点
        f1,f2=dk.gauss_solve_b(x,w,lamda,a,b,n)%利用Gauss_Legendre求解右端项
        %ff1,ff2=dk.quad_romberg(lamda,a,b,n)%利用quad和Romberg求解右端项
        A1,c1=dk.solve_AC(lamda,a,b,n,f1) %计算C
        A2,c2=dk.solve_AC(lamda,a,b,n,f2)
        error.append(dk.solve_xn(lamda,a,b,n,c1,c2)) %计算无穷范数
    error=pd.DataFrame(error)
    error.columns=['E1','E2']
    error.insert(0,'n',[i for i in range(1,nn+1,1)])
    print(error)
    plt.figure(2)
    plt.plot(error['n'],-np.log10(error['E1']),'o',label='error in x1')
    plt.plot(error['n'],-np.log10(error['E2']),'*',label='error in x2')
    plt.title('Figure2.(where a=0,b=1,lambda=5)')
    plt.xlabel('degree n')
    plt.ylabel('-log10 of error')
    plt.legend()
    print('all_cost_time:',time.time()-start_time)