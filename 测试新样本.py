# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:46:04 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

#submission = pd.read_csv('submission.csv',header=None, index_col=None)
res1 = pd.read_csv('./resnet/3new_sub99.3.csv',header=None, index_col=None)
res2 = pd.read_csv('./resnet/3res_vote_99.20.csv',header=None, index_col=None)
res3 = pd.read_csv('./resnet/5res_sub_99.3.csv',header=None, index_col=None)
#res4 = pd.read_csv('./resnet/resnet101_seed444_98.4.csv',header=None, index_col=None)
#res5 = pd.read_csv('./resnet/resnet152_333_submission98.4.csv',header=None, index_col=None)
#res6 = pd.read_csv('./resnet/resnet152_888_98.9_submission.csv',header=None, index_col=None)
#res7 = pd.read_csv('./resnet/resnet152_888_submission98.6.csv',header=None, index_col=None)
#res8 = pd.read_csv('./resnet/resnet156_666_submission.csv',header=None, index_col=None)

final = pd.DataFrame({ 0: res1[0], '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1]
                                   })       
final[4]=0                       
for i in range(1000):
    final[4][i] = Counter([final['1label'][i],
                           final['2label'][i],
                           final['3label'][i],
                          ]).most_common(1)[0][0]
    
result= final[[0,4]]
result.to_csv('./resnet/3best_sub.csv',index=False,header=False)
