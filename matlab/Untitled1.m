
Created on Wed Apr 11 14:17:52 2018

@author: shaowu

import degenerate_kernel as dk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
if  name__ == '__main__':
    start_time=time.time()
    print('******************************�������*******************************')
    %lamda=int(input('pleas input lambda:'))
    nn=int(input('please input degree n:'))
    %a=int(input('please input the left value of interval:'))
    %b=int(input('please input the right value of interval:'))
    %m=int(input('please input the node of Gauss_Legendre:'))
    %print('������...')
    lamda=5;a=0;b=1;m=500
    error=[]
    for n in range(1,nn+1,1):%��Χ��1��nn+1,�����1Ϊ����������ʡ��
        x,w=dk.gauss_xw(m) %ȡ400����˹��
        f1,f2=dk.gauss_solve_b(x,w,lamda,a,b,n)%����Gauss_Legendre����Ҷ���
        %ff1,ff2=dk.quad_romberg(lamda,a,b,n)%����quad��Romberg����Ҷ���
        A1,c1=dk.solve_AC(lamda,a,b,n,f1) %����C
        A2,c2=dk.solve_AC(lamda,a,b,n,f2)
        error.append(dk.solve_xn(lamda,a,b,n,c1,c2)) %���������
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