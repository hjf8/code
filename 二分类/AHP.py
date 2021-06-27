# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:14:25 2019

@author: 11876
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def get_w(array):
    row = array.shape[0]  # 计算出阶数
    a_axis_0_sum = array.sum(axis=0)
    # print(a_axis_0_sum)
    b = array / a_axis_0_sum  # 新的矩阵b
    # print(b)
    b_axis_0_sum = b.sum(axis=0)
    b_axis_1_sum = b.sum(axis=1)  # 每一行的特征向量
    # print(b_axis_1_sum)
    w = b_axis_1_sum / row  # 归一化处理(特征向量)
    nw = w * row
    AW = (w * array).sum(axis=1)
    # print(AW)
    max_max = sum(AW / (row * w))
    # print(max_max)
    CI = (max_max - row) / (row - 1)
    CR = CI / RI_dict[row]
    print(CR)
    if CR < 0.1:
        # print(round(CR, 3))
        print('满足一致性')
        # print(np.max(w))
        # print(sorted(w,reverse=True))
        # print(max_max)
        # print('特征向量:%s' % w)
        return w
    else:
        print(round(CR, 3))
        print('不满足一致性，请进行修改')


def main(array):
    if type(array) is np.ndarray:
        return get_w(array)
    else:
        print('请输入numpy对象')
#e=np.array([[1,6,3,5,4],[1/6,1,1/5,1/4,1/2],[1/3,5,1,2,3],[1/5,4,1/2,1,2],[1/4,2,1/3,1/2,1]])#第三问的判断矩阵
e = np.array([[1,5/4,4/6,2,3,4,9/2,5,6],[4/5,1,8/5,2,9/4,3,4,4,5],[6/4,5/8,1,6/5,2,5/2,3,4,5],
              [1/2,1/2,5/6,1,7/6,2,5/2,3,4],[1/3,4/9,1/2,6/7,1,5/4,2,5/2,3],[1/4,1/3,2/5,1/2,4/5,1,2,2,3],
              [2/9,1/4,1/3,2/5,1/2,1/2,1,3/2,2],[1/5,1/4,1/4,1/3,2/5,1/2,2/3,1,4/3],
              [1/6,1/5,1/5,1/4,1/3,1/3,1/2,3/4,1]])#第二问的判断矩阵
w_e=(main(e))#特征向量
x,y=np.linalg.eig(e)
  
