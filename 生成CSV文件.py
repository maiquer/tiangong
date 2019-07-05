# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:21:56 2018

@author: maiquer
"""
import os
import pandas as pd

path='./5/'
name_list =  []

for name in os.listdir(path):
    name_list.append(name)
label = 5    
aa = {'pic_name' : name_list, 'label' : label}
bb = pd.DataFrame(aa)
bb.to_csv('test5.csv',index=False,header=False)