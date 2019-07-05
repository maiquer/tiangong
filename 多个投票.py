# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:33:12 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:51:56 2018

@author: maiquer
"""

import numpy as np
import pandas as pd
from collections import Counter


res1 = pd.read_csv('./vote/vote_169_OHEM+OHEM0.7.csv',header=None, index_col=None)
res2 = pd.read_csv('./vote/vote161_3.csv',header=None, index_col=None)
res3 = pd.read_csv('./vote/vote161_single.csv',header=None, index_col=None)


res4 = pd.read_csv('./vote/vote169_single.csv',header=None, index_col=None)
res5 = pd.read_csv('./vote/vote169+OHEM0.8+cross512.csv',header=None, index_col=None)
res6 = pd.read_csv('./vote/vote201.csv',header=None, index_col=None)

#res7 = pd.read_csv('../99.5.csv',header=None, index_col=None)
#res8 = pd.read_csv('201-512-single-cross-666.csv',header=None, index_col=None)
#res9 = pd.read_csv('201-OHEM0.7-512-666.csv',header=None, index_col=None)
t = pd.read_csv('./B/100.csv',header=None, index_col=None)

final = pd.DataFrame({ 0: res1[0], '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1],
                                   '4label': res4[1], 
                                   '5label': res5[1], 
                                   '6label': res6[1], 
                                   'xlabel': t[1],       
                                   #'7label': res7[1],
                                   #'9label': res9[1],
                                   #'8label': res8[1]
                                   })




final[1]=0
for i in range(1000):
    final[1][i] = Counter([final['1label'][i],
                           final['2label'][i],
                           final['3label'][i],
                           final['4label'][i],
                           final['5label'][i],
                           final['6label'][i]],
    ).most_common(1)[0][0]
    
pic_name_list = []
label_list =[]
n = 0 
result= final[[0,1]]

for j in range(1000):
    if (final['1label'][j]==final['2label'][j]
    ==final['3label'][j]==final['4label'][j]
    ==final['5label'][j]==final['6label'][j]
#    ==final['xlabel'][j]
    ):
        pic_name_list.append(final[0][j])
        label_list.append(final['1label'][j])
        n=n+1
    else:
        final[1][j]=0   
result= final[[0,1]]
result.to_csv('./B/init_learn.csv',index=False,header=False)

sampel_train = list(zip(pic_name_list,label_list))
pretrain = pd.DataFrame(data = sampel_train)  
pretrain.to_csv('./vote/learnb.csv') 
a = pretrain[1].value_counts()     
print(n)
#result.to_csv('allvote.csv',index=False,header=False)

