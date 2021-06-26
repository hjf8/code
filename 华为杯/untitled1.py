# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:32:48 2020

@author: Sun
"""

s = input('请输入字符串:')
m1 = []
m2 = []
for i in range(len(s)):
    k = s[i]
    if k == '(':
        m1.append(i)
    if k == ')':
        m2.append(i)
l = m2[-1]-m1[0]        
print(l)   