# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:30:01 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

#submission = pd.read_csv('submission.csv',header=None, index_col=None)
res1 = pd.read_csv('./B/sam_res152_456_corss_2_97.8.csv',header=None, index_col=None)
res2 = pd.read_csv('./B/sam_res152_456_cross_3_97.2.csv',header=None, index_col=None)
res3 = pd.read_csv('./B/sam_res152_456_cross96.9.csv',header=None, index_col=None)

final = pd.DataFrame({ 0: res1[0], '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1]
                                   })

final[1]=0
for i in range(1000):
    final[1][i] = Counter([
                           final['1label'][i],
                           final['2label'][i],
                           final['3label'][i]                                             
                           ]).most_common(1)[0][0]
    
result= final[[0,1]]
result.to_csv('./B/sam.csv',index=False,header=False)
