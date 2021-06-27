# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:04:20 2019

@author: Junfeng Hu
"""

import numpy as np


print('2.a')
L = [1,2,3,4,4,5,7,7,7,10]#6,8,9美欧出现
a = np.array(L)
not_appear = []
for i in range(1,len(a)+1):
    if i not in a:
        not_appear.append(i)
        
print(not_appear)
    
print('2.b')
#x = int(input('请输入数字'))



input=60
if(input<2):
    print('The integer has no prime.')
num_prime=0 # statistic the number of prime
prime=[]
flag=True
for i in range(2,input+1):
    flag=True
    for j in range(2,int(np.sqrt(i))+1):
        if(i%j==0):
            flag=False
            break
    if flag:
        num_prime+=1
        prime.append(i)

p=[]
s=1
while(s < input):
    for k in range(len(prime)):
        if(input % prime[k]==0 and s< input):
            s*=prime[k]
            p.append(prime[k])
p= np.sort(p)
print('p:',p)
print('%d=%d*%d*%d*%d'%(input,p[0],p[1],p[2],p[3])  )