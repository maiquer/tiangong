# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:56:53 2018

@author: maiquer
"""

import numpy as np
import pandas as pd
from collections import Counter


best = pd.read_csv('best5_99.8.csv',header=None, index_col=None)
best1 = best.copy()
for i in range(0,430):
    best1[1][i]=0
for i in range(431,1000):
    best1[1][i]=0   
best1.to_csv('431.csv',index=False,header=False)



# 缩小正确的范围
#for i in range(425,450):
#    best2.drop([i],inplace=True)
#for i in range(650,675):
#    best2.drop([i],inplace=True)    
#    
#best2.to_csv('best2_950.csv',index=False,header=False)