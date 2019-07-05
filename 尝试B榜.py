# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:03:55 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:04:39 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

res1 = pd.read_csv('./B/Bnew_resenet152_456_OHEM0.7_256_96.5.csv',header=None, index_col=None)
res2 = pd.read_csv('./B/Bnew_resnet152_456_ohem0.7_256.csv',header=None, index_col=None)
res3 = pd.read_csv('./B/Bnew_restart_resnet152_456_cross.csv',header=None, index_col=None)

res4 = pd.read_csv('./B/Bnew_restart_resnet152_666_cross.csv',header=None, index_col=None)
res5 = pd.read_csv('./B/Bnew_se_resnet101_456_cross_256.csv',header=None, index_col=None)
res6 = pd.read_csv('./B/Bnew_se_resnet101_456_ohem0.7_256_96.csv',header=None, index_col=None)

res7 = pd.read_csv('./B/Bse_resnet152_456_cross_256.csv',header=None, index_col=None)
res8 = pd.read_csv('./B/Bse_resnet152_123_OHEM0.5_512_3.csv',header=None, index_col=None)

res9 = pd.read_csv('./B/Bse_resnet101_456_cross_256_96.9.csv',header=None, index_col=None)
res10 = pd.read_csv('./B/Bresnet152_666_cross_256.csv',header=None, index_col=None)
res11 = pd.read_csv('./B/Bresnet152_1113.csv',header=None, index_col=None)
res12 = pd.read_csv('./B/Bresenet152_456_OHEM0.7_256.csv',header=None, index_col=None)

t = pd.read_csv('./B/99.csv',header=None, index_col=None)
final = pd.DataFrame({ 0: res9[0], 
                                   '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1],
                                   '4label': res4[1],
                                   '5label': res5[1], 
                                   '6label': res6[1],
#                                   '7label': res7[1],
#                                   '8label': res8[1],
#                                   '9label': res9[1],
#                                   '10label': res10[1],
#                                   '11label': res11[1],
#                                   '12label': res12[1],
                                   'xlabel': t[1],                       
                                   })

final[1]=0
for i in range(1000):
    final[1][i] = Counter([
                           final['1label'][i],
                           final['2label'][i],
                           final['3label'][i],
                           final['4label'][i],
                           final['5label'][i],
                           final['6label'][i],
#                           final['7label'][i],
#                           final['8label'][i],
#                           final['9label'][i],
#                           final['10label'][i],
#                           final['11label'][i],
#                           final['12label'][i],
                           final['xlabel'][i],
                           ]).most_common(1)[0][0]
    
result= final[[0,1]]
result.to_csv('./B/5.csv',index=False,header=False)
