# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:18:50 2019

@author: Junfeng Hu
"""

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
print('%d=%d*%d*%d*%d'%(input,p[0],p[1],p[2],p[3])  )
'''
x=60
def factors(x):
    nums = []
    num = 1
    for num in range(2,int(np.sqrt(x)+1)):
        if x%num == 0: 
            nums.append(num)
    return nums
nums = factors(x)
i = nums[0]
i0 = x/i
j = factors(i0)
j1 = int(j[0])
j0 = int(i0/j[0])
k = factors(j0)
k0 = int(k[0])
k1 = int(j0/k[0])
print('%d=%d*%d*%d*%d'%(x,i,j1,k0,k1)  )
'''