# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:41:34 2019

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

data=pd.read_csv('PMD20181122002FL00LJ.csv',usecols=['脉搏'], encoding = 'gb18030')

ed
'''
#EMD经验模态分解
decomposer = EMD(data['脉搏'])
imfs = decomposer.decompose()
#绘制分解图
plot_imfs(data['脉搏'],imfs,data.index)

#保存IMFs
arr = np.vstack((imfs,data['脉搏']))
dataframe = pd.DataFrame(arr.T)
dataframe.to_csv('D:/imf.csv',index=None,columns=None)
dataframe['和'] = (dataframe[1]+dataframe[2]+dataframe[3]+dataframe[4]+dataframe[5]+dataframe[6]+dataframe[7]+dataframe[8])/8
#plt.plot(dataframe['和'][0:300])
'''

#data = data[:-1]
#wavelet = pywt.Wavelet('db1')
s = np.zeros(128)
cA, cD = pywt.dwt(data['脉搏'], 'db4')#0-50,50-100


#plt.plot(data['脉搏'])
cA1, cA2 = pywt.dwt(cA, 'db4')#0-25, 25-50
#cA1
cA11, cA12 = pywt.dwt(cA1, 'db4')#0-12.5, 12.5-25


cA111, cA112 = pywt.dwt(cA11, 'db4')#0-6.25, 6.25-12.5

cA1111, cA1112 = pywt.dwt(cA111, 'db4')#0-3.125, 3.125-6.25
cA11111, cA11112 = pywt.dwt(cA1111, 'db4')#0-1.5625, 1.5625-3.125
cA111111, cA111112 = pywt.dwt(cA11111, 'db4')#0-0.78125, 0.78125-1.5625***********
cA111121, cA111122 = pywt.dwt(cA11112, 'db4')#1.5625-2.34375, 2.34375-3.125*******
'''
s[0] = abs(cA111111).sum()
s[1] = abs(cA111112).sum()
s[2] = abs(cA111121).sum()
s[3] = abs(cA111122).sum()
cA11121, cA11122 = pywt.dwt(cA1112, 'db4')#3.125-4.6875, 4.6875-6.25
cA111211, cA111212 = pywt.dwt(cA11121, 'db4')#3.125-3.90605, 3.90625-4.6875********
cA111221, cA111222 = pywt.dwt(cA11122, 'db4')#4.6875-5.46875, 5.46875-6.25*********
s[4] = abs(cA111211).sum()
s[5] = abs(cA111212).sum()
s[6] = abs(cA111221).sum()
s[7] = abs(cA111222).sum()

cA1121, cA1122 = pywt.dwt(cA112, 'db4')#6.25-9.375, 9.375-12.5
cA11211, cA11212 = pywt.dwt(cA1121, 'db4')#6.25-7.8125, 7.8125-9.375
cA112111, cA112112 = pywt.dwt(cA11211, 'db4')#6.25-7.03125, 7.03125-7.8125***********
cA112121, cA112122 = pywt.dwt(cA11212, 'db4')#7.8125-8.59375, 8.59375-9.375**********
s[8] = abs(cA112111).sum()
s[9] = abs(cA112112).sum()
s[10] = abs(cA112121).sum()
s[11] = abs(cA112122).sum()
cA11221, cA11222 = pywt.dwt(cA1122, 'db4')#9.375-10.9375, 10.9375-12.5
cA112211, cA112212 = pywt.dwt(cA11221, 'db4')#9.375-10.15625, 10.15625-10.9375********
cA112221, cA112222 = pywt.dwt(cA11222, 'db4')#10.9375-11.71875, 11.71875-12.5*********
s[12] = abs(cA112211).sum()
s[13] = abs(cA112212).sum()
s[14] = abs(cA112221).sum()
s[15] = abs(cA112222).sum()


cA121, cA122 = pywt.dwt(cA12, 'db4')#12.5-18.75, 18.75-25

cA1211, cA1212 = pywt.dwt(cA121, 'db4')#12.5-15.625, 15.625-18.75
cA12111, cA12112 = pywt.dwt(cA1211, 'db4')#12.5-14.0625, 14.0625-15.625
cA121111, cA121112 = pywt.dwt(cA12111, 'db4')#125.-13.28125, 13.28125-14.0625***********
cA121121, cA121122 = pywt.dwt(cA12112, 'db4')#14.0625-14.84375, 14.84375-15.625*******
s[16] = abs(cA121111).sum()
s[17] = abs(cA121112).sum()
s[18] = abs(cA121121).sum()
s[19] = abs(cA121122).sum()
cA12121, cA12122 = pywt.dwt(cA1212, 'db4')#15.625-17.1875, 17.1875-18.75
cA121211, cA121212 = pywt.dwt(cA12121, 'db4')#15.625-16.40625, 16.40625-17.1875********
cA121221, cA121222 = pywt.dwt(cA12122, 'db4')#17.1875-17.96875, 17.96875-18.75-*********
s[20] = abs(cA121211).sum()
s[21] = abs(cA121212).sum()
s[22] = abs(cA121221).sum()
s[23] = abs(cA121222).sum()

cA1221, cA1222 = pywt.dwt(cA122, 'db4')#18.75-21.875, 21.875-25
cA12211, cA12212 = pywt.dwt(cA1221, 'db4')#18.75-20.3125, 20.3125-21.875
cA122111, cA122112 = pywt.dwt(cA12211, 'db4')#18.75-19.53125, 19.53125-20.3125***********
cA122121, cA122122 = pywt.dwt(cA12212, 'db4')#20.3125-21.09375, 21.09375-21.875**********
s[24] = abs(cA122111).sum()
s[25] = abs(cA122112).sum()
s[26] = abs(cA122121).sum()
s[27] = abs(cA122122).sum()
cA12221, cA12222 = pywt.dwt(cA1222, 'db4')#21.875-23.4375, 23.4375-25
cA122211, cA122212 = pywt.dwt(cA12221, 'db4')#21.875-22.65625, 22.65625-23.4375********
cA122221, cA122222 = pywt.dwt(cA12222, 'db4')#23.4375-24.21875, 24.21875-25*********
s[28] = abs(cA122211).sum()
s[29] = abs(cA122212).sum()
s[30] = abs(cA122221).sum()
s[31] = abs(cA122222).sum()


#cA2
cA21, cA22 = pywt.dwt(cA2, 'db4')#25-37.5,37.5-50

cA211, cA212 = pywt.dwt(cA21, 'db4')

cA2111, cA2112 = pywt.dwt(cA211, 'db4')
cA21111, cA21112 = pywt.dwt(cA2111, 'db4')
cA211111, cA211112 = pywt.dwt(cA21111, 'db4')#***********
cA211121, cA211122 = pywt.dwt(cA21112, 'db4')#***********
s[32] = abs(cA211111).sum()
s[33] = abs(cA211112).sum()
s[34] = abs(cA211121).sum()
s[35] = abs(cA211122).sum()
cA21121, cA21122 = pywt.dwt(cA2112, 'db4')
cA211211, cA211212 = pywt.dwt(cA21121, 'db4')#********
cA211221, cA211222 = pywt.dwt(cA21122, 'db4')#*********
s[36] = abs(cA211211).sum()
s[37] = abs(cA211212).sum()
s[38] = abs(cA211221).sum()
s[39] = abs(cA211222).sum()

cA2121, cA2122 = pywt.dwt(cA212, 'db4')
cA21211, cA21212 = pywt.dwt(cA2121, 'db4')
cA212111, cA212112 = pywt.dwt(cA21211, 'db4')#***********
cA212121, cA212122 = pywt.dwt(cA21212, 'db4')#**********
s[40] = abs(cA212111).sum()
s[41] = abs(cA212112).sum()
s[42] = abs(cA212121).sum()
s[43] = abs(cA212122).sum()
cA21221, cA21222 = pywt.dwt(cA2122, 'db4')
cA212211, cA212212 = pywt.dwt(cA21221, 'db4')#********
cA212221, cA212222 = pywt.dwt(cA21222, 'db4')#*********
s[44] = abs(cA212211).sum()
s[45] = abs(cA212212).sum()
s[46] = abs(cA212221).sum()
s[47] = abs(cA212222).sum()


cA221, cA222 = pywt.dwt(cA22, 'db4')

cA2211, cA2212 = pywt.dwt(cA221, 'db4')
cA22111, cA22112 = pywt.dwt(cA2211, 'db4')
cA221111, cA221112 = pywt.dwt(cA22111, 'db4')#***********
cA221121, cA221122 = pywt.dwt(cA22112, 'db4')#***********
s[48] = abs(cA221111).sum()
s[49] = abs(cA221112).sum()
s[50] = abs(cA221121).sum()
s[51] = abs(cA221122).sum()
cA22121, cA22122 = pywt.dwt(cA2212, 'db4')
cA221211, cA221212 = pywt.dwt(cA22121, 'db4')#********
cA221221, cA221222 = pywt.dwt(cA22122, 'db4')#*********
s[52] = abs(cA221211).sum()
s[53] = abs(cA221212).sum()
s[54] = abs(cA221221).sum()
s[55] = abs(cA221222).sum()
'''
'''
cA2221, cA2222 = pywt.dwt(cA222, 'db2')
cA22211, cA22212 = pywt.dwt(cA2221, 'db2')
cA222111, cA222112 = pywt.dwt(cA22211, 'db2')#**********
cA222121, cA222122 = pywt.dwt(cA22212, 'db2')#**********
s[56] = abs(cA222111).sum()
s[57] = abs(cA222112).sum()
s[58] = abs(cA222121).sum()
s[59] = abs(cA222122).sum()
cA22221, cA22222 = pywt.dwt(cA2222, 'db2')
cA222211, cA222212 = pywt.dwt(cA22221, 'db2')#*********
cA222221, cA222222 = pywt.dwt(cA22222, 'db2')#*********
s[60] = abs(cA222211).sum()
s[61] = abs(cA222212).sum()
s[62] = abs(cA222221).sum()
s[63] = abs(cA222222).sum()


cD1, cD2 = pywt.dwt(cD, 'db2')#0-25, 25-50
#cA1
cD11, cD12 = pywt.dwt(cD1, 'db2')#0-12.5, 12.5-25

cD111, cD112 = pywt.dwt(cD11, 'db2')#0-6.25, 6.25-12.5

cD1111, cD1112 = pywt.dwt(cD111, 'db2')#0-3.125, 3.125-6.25
cD11111, cD11112 = pywt.dwt(cD1111, 'db2')#0-1.5625, 1.5625-3.125
cD111111, cD111112 = pywt.dwt(cD11111, 'db2')#0-0.78125, 0.78125-1.5625***********
cD111121, cD111122 = pywt.dwt(cD11112, 'db2')#1.5625-2.34375, 2.34375-3.125*******
s[64] = abs(cD111111).sum()
s[65] = abs(cD111112).sum()
s[66] = abs(cD111121).sum()
s[67] = abs(cD111122).sum()
cD11121, cD11122 = pywt.dwt(cD1112, 'db2')#3.125-4.6875, 4.6875-6.25
cD111211, cD111212 = pywt.dwt(cD11121, 'db2')#3.125-3.90605, 3.90625-4.6875********
cD111221, cD111222 = pywt.dwt(cD11122, 'db2')#4.6875-5.46875, 5.46875-6.25*********
s[68] = abs(cD111211).sum()
s[69] = abs(cD111212).sum()
s[70] = abs(cD111221).sum()
s[71] = abs(cD111222).sum()

cD1121, cD1122 = pywt.dwt(cD112, 'db2')#6.25-9.375, 9.375-12.5
cD11211, cD11212 = pywt.dwt(cD1121, 'db2')#6.25-7.8125, 7.8125-9.375
cD112111, cD112112 = pywt.dwt(cD11211, 'db2')#6.25-7.03125, 7.03125-7.8125***********
cD112121, cD112122 = pywt.dwt(cD11212, 'db2')#7.8125-8.59375, 8.59375-9.375**********
s[72] = abs(cD112111).sum()
s[73] = abs(cD112112).sum()
s[74] = abs(cD112121).sum()
s[75] = abs(cD112122).sum()
cD11221, cD11222 = pywt.dwt(cD1122, 'db2')#9.375-10.9375, 10.9375-12.5
cD112211, cD112212 = pywt.dwt(cD11221, 'db2')#9.375-10.15625, 10.15625-10.9375********
cD112221, cD112222 = pywt.dwt(cD11222, 'db2')#10.9375-11.71875, 11.71875-12.5*********
s[76] = abs(cD112211).sum()
s[77] = abs(cD112212).sum()
s[78] = abs(cD112221).sum()
s[79] = abs(cD112222).sum()


cD121, cD122 = pywt.dwt(cD12, 'db2')#12.5-18.75, 18.75-25

cD1211, cD1212 = pywt.dwt(cD121, 'db2')#12.5-15.625, 15.625-18.75
cD12111, cD12112 = pywt.dwt(cD1211, 'db2')#12.5-14.0625, 14.0625-15.625
cD121111, cD121112 = pywt.dwt(cD12111, 'db2')#125.-13.28125, 13.28125-14.0625***********
cD121121, cD121122 = pywt.dwt(cD12112, 'db2')#14.0625-14.84375, 14.84375-15.625*******
s[80] = abs(cD121111).sum()
s[81] = abs(cD121112).sum()
s[82] = abs(cD121121).sum()
s[83] = abs(cD121122).sum()
cD12121, cD12122 = pywt.dwt(cD1212, 'db2')#15.625-17.1875, 17.1875-18.75
cD121211, cD121212 = pywt.dwt(cD12121, 'db2')#15.625-16.40625, 16.40625-17.1875********
cD121221, cD121222 = pywt.dwt(cD12122, 'db2')#17.1875-17.96875, 17.96875-18.75-*********
s[84] = abs(cD121211).sum()
s[85] = abs(cD121212).sum()
s[86] = abs(cD121221).sum()
s[87] = abs(cD121222).sum()

cD1221, cD1222 = pywt.dwt(cD122, 'db2')#18.75-21.875, 21.875-25
cD12211, cD12212 = pywt.dwt(cD1221, 'db2')#18.75-20.3125, 20.3125-21.875
cD122111, cD122112 = pywt.dwt(cD12211, 'db2')#18.75-19.53125, 19.53125-20.3125***********
cD122121, cD122122 = pywt.dwt(cD12212, 'db2')#20.3125-21.09375, 21.09375-21.875**********
s[88] = abs(cD122111).sum()
s[89] = abs(cD122112).sum()
s[90] = abs(cD122121).sum()
s[91] = abs(cD122122).sum()
cD12221, cD12222 = pywt.dwt(cD1222, 'db2')#21.875-23.4375, 23.4375-25
cD122211, cD122212 = pywt.dwt(cD12221, 'db2')#21.875-22.65625, 22.65625-23.4375********
cD122221, cD122222 = pywt.dwt(cD12222, 'db2')#23.4375-24.21875, 24.21875-25*********
s[92] = abs(cD122211).sum()
s[93] = abs(cD122212).sum()
s[94] = abs(cD122221).sum()
s[95] = abs(cD122222).sum()


#cA2
cD21, cD22 = pywt.dwt(cD2, 'db2')#25-37.5,37.5-50

cD211, cD212 = pywt.dwt(cD21, 'db2')

cD2111, cD2112 = pywt.dwt(cD211, 'db2')
cD21111, cD21112 = pywt.dwt(cD2111, 'db2')
cD211111, cD211112 = pywt.dwt(cD21111, 'db2')#***********
cD211121, cD211122 = pywt.dwt(cD21112, 'db2')#***********
s[96] = abs(cD211111).sum()
s[97] = abs(cD211112).sum()
s[98] = abs(cD211121).sum()
s[99] = abs(cD211122).sum()
cD21121, cD21122 = pywt.dwt(cD2112, 'db2')
cD211211, cD211212 = pywt.dwt(cD21121, 'db2')#********
cD211221, cD211222 = pywt.dwt(cD21122, 'db2')#*********
s[100] = abs(cD211211).sum()
s[101] = abs(cD211212).sum()
s[102] = abs(cD211221).sum()
s[103] = abs(cD211222).sum()

cD2121, cD2122 = pywt.dwt(cD212, 'db2')
cD21211, cD21212 = pywt.dwt(cD2121, 'db2')
cD212111, cD212112 = pywt.dwt(cD21211, 'db2')#***********
cD212121, cD212122 = pywt.dwt(cD21212, 'db2')#**********
s[104] = abs(cD212111).sum()
s[105] = abs(cD212112).sum()
s[106] = abs(cD212121).sum()
s[107] = abs(cD212122).sum()
cD21221, cD21222 = pywt.dwt(cD2122, 'db2')
cD212211, cD212212 = pywt.dwt(cD21221, 'db2')#*********
cD212221, cD212222 = pywt.dwt(cD21222, 'db2')#*********
s[108] = abs(cD212211).sum()
s[109] = abs(cD212212).sum()
s[110] = abs(cD212221).sum()
s[111] = abs(cD212222).sum()

cD221, cD222 = pywt.dwt(cD22, 'db2')

cD2211, cD2212 = pywt.dwt(cD221, 'db2')
cD22111, cD22112 = pywt.dwt(cD2211, 'db2')
cD221111, cD221112 = pywt.dwt(cD22111, 'db2')#***********
cD221121, cD221122 = pywt.dwt(cD22112, 'db2')#***********
s[112] = abs(cD221111).sum()
s[113] = abs(cD221112).sum()
s[114] = abs(cD221121).sum()
s[115] = abs(cD221122).sum()
cD22121, cD22122 = pywt.dwt(cD2212, 'db2')
cD221211, cD221212 = pywt.dwt(cD22121, 'db2')#********
cD221221, cD221222 = pywt.dwt(cD22122, 'db2')#*********
s[116] = abs(cD221211).sum()
s[117] = abs(cD221212).sum()
s[118] = abs(cD221221).sum()
s[119] = abs(cD221222).sum()

cD2221, cD2222 = pywt.dwt(cD222, 'db2')
cD22211, cD22212 = pywt.dwt(cD2221, 'db2')
cD222111, cD222112 = pywt.dwt(cD22211, 'db2')#**********
cD222121, cD222122 = pywt.dwt(cD22212, 'db2')#**********
s[120] = abs(cD222111).sum()
s[121] = abs(cD222112).sum()
s[122] = abs(cD222121).sum()
s[123] = abs(cD222122).sum()
cD22221, cD22222 = pywt.dwt(cD2222, 'db2')
cD222211, cD222212 = pywt.dwt(cD22221, 'db2')#*********
cD222221, cD222222 = pywt.dwt(cD22222, 'db2')#*********
s[124] = abs(cD222211).sum()
s[125] = abs(cD222212).sum()
s[126] = abs(cD222221).sum()
s[127] = abs(cD222222).sum()
'''
r1 = np.zeros(30)
r2 = np.zeros(54)
r3 = np.zeros(101)
r4 = np.zeros(195)
r5 = np.zeros(383)
r6 = np.zeros(760)
r7 = np.zeros(1513)


m1 = pywt.idwt(cA111111,r1, 'db4')
m2 = pywt.idwt(m1,r2, 'db4')
m2 = m2[:-1]
m3 = pywt.idwt(m2,r3, 'db4')
m3 = m3[:-1] 
m4 = pywt.idwt(m3,r4, 'db4')
m4 = m4[:-1]
m5 = pywt.idwt(m4,r5, 'db4')
#m5 = m5[:-1]
m6 = pywt.idwt(m5,r6, 'db4')
m6 = m6[:-1]
m7 = pywt.idwt(m6,r7, 'db4')
plt.plot(m7)