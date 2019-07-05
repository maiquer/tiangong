# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:52:24 2018

@author: maiquer
"""

import matplotlib.pyplot as plt
#import xgboost as xgb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


train = pd.read_csv('new_train.csv')
train = train.values
#test  = test.values
##所有的分类器模型,采用默认参数
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(probability=True),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    #xgb.XGBClassifier()
    ]

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)#生成十组训练集和测试集，每组测试集为1/10

x = train[:, 2:]
y = train[:, 1]
accuracy = np.zeros(len(classifiers))#每个模型的准确率
for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_num = 0
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        clf.fit(x_train, y_train)
        accuracy[clf_num] += (y_test == clf.predict(x_test)).mean()#该模型的准确率，十次平均
        clf_num += 1

accuracy = accuracy / 10
plt.bar(np.arange(len(classifiers)), accuracy, width=0.5, color='b')
plt.xlabel('Alog')  
plt.ylabel('Accuracy')  
#plt.xticks(np.arange(len(classifiers)) + 0.25, 
#           ('KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC', 'GNB',
#            'LDA', 'QDA', 'LR', 'xgb')
plt.xticks(np.arange(len(classifiers)) + 0.25, 
           ('KNN', 'DT', 'RF', 'SVC', 'AdaB', 'GBC','LDA', 'QDA', 'LR', 'xgb'))
print(accuracy)