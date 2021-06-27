# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:36:26 2019

@author: 11876
"""

import pywt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import scipy.io

#def xiaobo(data):
    
    
data=pd.read_csv('PMD20181121001ML00WS.csv',usecols=['脉搏'], encoding = 'gb18030')