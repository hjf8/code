# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:22:50 2018

@author: Junfeng Hu
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels