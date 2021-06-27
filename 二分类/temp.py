# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pywt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('PMD20181122002FL00LJ.csv',usecols=['脉搏'], encoding = 'gb18030')
#data = data[:-1]
#wavelet = pywt.Wavelet('db1')
cA, cD = pywt.dwt(data['脉搏'], 'db2')
#d=pywt.idwt(cA, cD,'db2')

#d=(pywt.idwt(cA, cD,'db2'))
#plt.plot(cA1)
#plt.plot(data['脉搏'])
cAA,cAD = pywt.dwt(cA,'db2')
cDA,cDD = pywt.dwt(cD,'db2')

cAAA,cAAD = pywt.dwt(cAA,'db2')
cADA,cADD = pywt.dwt(cAD,'db2')
#d=pywt.idwt(cAAA,cAAD,'db2')

cADAA,cADAD = pywt.dwt(cADA,'db2')

cADADA,cADADD = pywt.dwt(cADAD,'db2')

s1 = np.zeros((97))#cADADD
s2 = np.zeros((380))#cADD
s3 = np.zeros((380))#cAAA
s4 = np.zeros((1511))#cD
s5 = np.zeros((757))#cDA

r1 = pywt.idwt(cADADA, s1,'db2')
r1 = r1[:-1]
r2 = pywt.idwt(cADAA, r1,'db2')#cADA
r3 = pywt.idwt(r2, s2,'db2')#cAD
r3 = r3[:-1]
r4 = pywt.idwt(s3, cAAD,'db2')#cAA
r4 = r4[:-1]
r5 = pywt.idwt(r4, r3,'db2')
r5 = r5[:-1]
r6 = pywt.idwt(r5, s4,'db2')
#plt.plot(r6[0:500])

cDDA,cDDD = pywt.dwt(cDD,'db2')
r7 = pywt.idwt(s2, cDDD,'db2')
r7 = r7[:-1]#cDD
r8 = pywt.idwt(s5, r7,'db2')#cD
r8 = r8[:-1]
r9 = pywt.idwt(s4, r8,'db2')
#plt.plot(r9[0:80])

r10 = data['脉搏']-r6-r9
plt.plot(r10[0:500])


'''
ed
print(wavelet)
print(wavelet.dec_lo, wavelet.dec_hi)#分解滤波器值
print(wavelet.rec_lo, wavelet.rec_hi)#重构滤波器值
'''

