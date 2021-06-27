# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:49:07 2019

cur = conn.cursor()  
cur.execute("SELECT * FROM bigtable");  
row = cur.fetchone()  
while row is not None:  
#    do something  
    row = cur.fetchone()  
  
cur.close()  
conn.close() 
@author: 11876
"""
import pandas as pd
import numpy as np
x=[[1,2,3],[4,5,6]]
xx=pd.DataFrame(x)
xxx=np.array(xx[0].values)
y=xx.sum(axis=1)
