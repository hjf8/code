  # -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:28:54 2019

@author: Junfeng Hu
"""

#!/usr/bin/python3

print("The first test!")
m=[1,2,3,4]
for i in m:
	for j in m:
		for k in m:
			if (i!=j) & (i!=k) & (j!=k):
				print(str(i)+str(j)+str(k),end=' ')
				
print('\nThe second test!')
import numpy as np
B=np.random.rand(3,3)*5
b=np.random.rand(3,1)*5
print(np.matrix(B)*b)

print('The sort for three number')
m=[5,6,7]
print(np.sort(m))

print('The fourth test!')
for i in range(1,10):
	for j in range(1,i+1):
		print(str(i)+'*'+str(j)+'='+str(i*j),end=' ')
	print('')
	
print('The fifth test')
m='1562bnc84ij'
num=0
letter=0
for i in m:
	if i in [str(e) for e in range(10)]:
		num+=1
	else:
		letter+=1
print('num:%d,letter:%d'%(num,letter))

print('1!+2!+...+n!')
def jiecheng(n):
	if n==1:
		return 1
	else:
		return n*jiecheng(n-1)
s=0
m=5
for i in range(1,m):
	s+=jiecheng(i)
print('1!+...+%d!=%d'%(m,s))

print('The seventh test!')
m='yhgbkk12kj'
print(m[::-1])