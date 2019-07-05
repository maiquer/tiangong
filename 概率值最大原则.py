# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:26:57 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:25:23 2018

@author: maiquer
"""

import pandas as pd
import numpy as np

res_888 = np.load('./result/resnet152_888_98.9_prob.npy')
res_666 = np.load('./result/resnet156_666_submission_probability.npy')
res_111 = np.load('./result/resnet152_111_prob.npy')

label = []
t888 = 0
t666 = 0
t111 = 0
for i in range(len(res_888)):
    res888_prelabel = np.argmax(res_888[i])
    res888_probability = res_888[i][np.argmax(res_888[i])]
    
    res666_prelabel = np.argmax(res_666[i])
    res666_probability = res_666[i][np.argmax(res_666[i])]
    
    res111_prelabel = np.argmax(res_111[i])
    res111_probability = res_111[i][np.argmax(res_111[i])]
    
    if(res888_probability == max(res888_probability,res666_probability,res111_probability)):
        t888 = t888 + 1
        label.append(res888_prelabel)
        continue
    if(res666_probability == max(res888_probability,res666_probability,res111_probability)):
        t666 = t666 + 1
        label.append(res666_prelabel)
        continue
    if(res111_probability == max(res888_probability,res666_probability,res111_probability)):
        t111 = t111 + 1
        label.append(res111_prelabel)
        continue

print('\n')   
print('res888',t888)
print('res666',t666) 
print('res111',t111)     

# 转换成提交样例中的格式
result = pd.read_csv('./result/resnet152_888_98.9_prob.csv')
sub_filename, sub_label = [], []
for i in label:
    if i == 0:
        sub_label.append('OCEAN')
    if i == 1:
        sub_label.append('MOUNTAIN')
    if i == 2:
        sub_label.append('LAKE')
    if i == 3:
        sub_label.append('FARMLAND')
    if i == 4:
        sub_label.append('DESERT')
    if i == 5:
        sub_label.append('CITY')
for index, row in result.iterrows():
    sub_filename.append(row['filename'])
# 生成结果文件，保存在result文件夹中，可用于直接提交
submission = pd.DataFrame({'filename': sub_filename, 'label': sub_label})
submission.to_csv('3submission.csv', header=None, index=False)
