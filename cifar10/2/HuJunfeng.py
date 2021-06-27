# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#one
print('第一题结果：')
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if (i!=j)&(i!=k)&(k!=j):
                print('%d%d%d'%(i,j,k))

#two
print('第二题结果：')
#A = input('请输入一个矩阵：')
#b = input('请输入一个向量：')
A = np.random.rand(3,3)*10
b = np.random.rand(3,1)*10
A = np.mat(A)
b = np.mat(b)
print(A.T*b)

#three

print('第三题结果：')
def sort(a,b,c):
    if a>b:
        a,b = b,a
    if b>c:
        b,c = c,b
    if a>b:
        a,b = b,a
    return a,b,c
print(sort(3,1,2))

#four
print('第四题结果：')
def mul():
    for i in range(1,10):
        for j in range(1,i+1):
            print('%d*%d = %d'%(i,j,i*j),end = ' ')
        print(' ')
mul()

#five
print('第五题结果：')
def count(a):
    num = 0
    letter = 0
    for i in range(len(a)):
        for i in a[i]:
            if i.isalpha():
                letter +=1
            elif i.isnumeric():
                num +=1
    return num,letter
print(count('12abc123'))

#six
print('第六题结果：')
def jiecheng(n):
    if n == 1:
        return 1
    else:
        return n*jiecheng(n-1)
sum = 0
for i in range(1,21):
    sum += jiecheng(i)
print(sum)

#seven
print('第七题结果：')
def reverse(s):
    if len(s)<1:
        return s
    return reverse(s[1:])+s[0]
print(reverse('abcd'))
