import numpy as np
import pandas as pd
from collections import Counter

#submission = pd.read_csv('submission.csv',header=None, index_col=None)
res152_666 = pd.read_csv('./result/resnet156_666_submission.csv',header=None, index_col=None)
res152_888 = pd.read_csv('./result/resnet152_888_98.9_submission.csv',header=None, index_col=None)
res152_111 = pd.read_csv('./result/resnet152_111_submission.csv',header=None, index_col=None)
res152_333 = pd.read_csv('./result/resnet_333_submission.csv',header=None, index_col=None)
#oriv4 = pd.read_csv('./pro/v4_rand666_94.9_result.csv',header=None, index_col=None)

#des121 = pd.read_csv('./pro/desnet121_rand999_95.7_result.csv',header=None, index_col=None)
#res666 = pd.read_csv('./pro/resnet_rand666_95.37_result.csv',header=None, index_col=None)
#v4 = pd.read_csv('./pro/v4_rand777_94.55_result.csv',header=None, index_col=None)

tmp0 = pd.merge(res152_666,res152_888,how='left',on=[0])
tmp = pd.merge(tmp0,res152_111,how='left',on=[0])
final = pd.merge(tmp,res152_333,how='left',on=[0],suffixes=('_2', '_3'))
final[5]=0
#final[4].map(lambda x: Counter([final[1],final['1_x'],final['1_y']]).most_common(1)[0][0])
for i in range(1000):
    final[5][i] = Counter([final['1_2'][i],final['1_3'][i],final['1_x'][i],final['1_y'][i]]).most_common(1)[0][0]
    
result= final[[0,5]]
result.to_csv('./result/4sub.csv',index=False,header=False)