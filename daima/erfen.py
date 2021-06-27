# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:39:19 2020

@author: Sun
"""

def bin_search(data_set,val):
    #low 和high代表下标 最小下标，最大下标
    low=0
    high=len(data_set)-1
    while low <=high:# 只有当low小于High的时候证明中间有数
        mid=(low+high)//2
        if data_set[mid]==val:
            return mid  #返回他的下标
        elif data_set[mid]>val:
            high=mid-1
        else:
            low=mid+1
    return # return null证明没有找到
data_set = list(range(100))
print(bin_search(data_set, 9))
    


def function1(lists,k):
#   冒泡法
    length = len(lists)
    for i in range(k):
        for j in range(i+1,length):
            if lists[i] > lists[j]:
                lists[j],lists[i] = lists[i],lists[j]    
    
    
    
    
    
    
    
    
    
    
    
    
    