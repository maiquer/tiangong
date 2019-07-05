import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.sampler import  WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import  WeightedRandomSampler
from imblearn.over_sampling import SMOTE,RandomOverSampler
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import csv
from cnn_finetune import make_model
from torch.autograd import Variable
import torch.nn.functional as F

def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')

class TrainDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader= default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row[0], row[2]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    
train_path = './train/'
test_path= './test/'
testb_path= './testB/'

label2num= {
        'OCEAN':0,
        'MOUNTAIN':1,
        'LAKE':2,
        'FARMLAND':3,
        'DESERT':4,
        'CITY':5
    }

all_csv = pd.read_csv('000.csv',header = None,encoding='utf-8')
all_csv[0] = all_csv[0].map(lambda x:train_path+x)
ori_label = all_csv[1]
all_csv[2] = all_csv[1].map(lambda x:label2num[x])
all_csv = all_csv.drop([1], axis=1)

#test_csv = pd.read_csv('test_result_97.csv',header = None,encoding='utf-8')
#test_csv[0] = test_csv[0].map(lambda x:test_path+x)
#temp_label = test_csv[1]
#test_csv[2] = test_csv[1].map(lambda x:label2num[x])
#test_csv = test_csv.drop([1], axis=1)
#
#testb_csv = pd.read_csv('pre_testb.csv',header = None,encoding='utf-8')
#testb_csv[0] = testb_csv[0].map(lambda x:testb_path+x)
#testb_csv[2] = testb_csv[1].map(lambda x:label2num[x])
#testb_csv = testb_csv.drop([1], axis=1)

#result = train_csv.append(test_csv)
#result = result.append(testb_csv)
#
#ocean = result[result[2] == 0]
#ocean = ocean.head(456) 
#
#mountain = result[result[2] == 1]
#mountain = mountain.head(464) 
#
#lake = result[result[2] == 2]
#farmland = result[result[2] == 3]
#
#desert = result[result[2] == 4]
#desert = desert.head(636)
#
#city = result[result[2] == 5]
#
#all_label = ocean.append(mountain)
#all_label = all_label.append(lake)
#all_label = all_label.append(farmland)
#all_label = all_label.append(desert)
#all_label = all_label.append(city)

train_csv, val_csv = train_test_split(all_csv, test_size=0.1, random_state=123, stratify = all_csv[2])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_data = TrainDataset(train_csv,
          transform=transforms.Compose([
              transforms.Resize((256, 256)),
              transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
              transforms.RandomHorizontalFlip(),
              transforms.RandomVerticalFlip(),
              #transforms.RandomCrop(224),
              transforms.ToTensor(),
              normalize,
          ]))
train_loader = DataLoader(train_data, batch_size = 30)
a = np.zeros((len(train_csv),3,256,256))
d = []
for i, (images, target) in enumerate(train_loader):
    a[30*i:30*(i+1)] = np.array(images)
    b = np.array(target)
    d.append(b)
oversampler = RandomOverSampler ( random_state = 666 )
x_train = list(a)
os_features,os_labels=oversampler.fit_sample( a, d)

#weight_per_class = [0.044, 0.236, 0.2, 0.329, 0.065, 0.128]
#def make_weights_for_balanced_classes(train_data):                                                   
#    weight = [0] * len(train_data)  
#    i = 0                                            
#    for images, target in enumerate(train_data):                                     
#        weight[i] = weight_per_class[target[1]]     
#        i = i+1                             
#    return weight
#
#classesid = [0,1,2,3,4,5,6]
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#train_data = TrainDataset(train_csv,
#          transform=transforms.Compose([
#              transforms.Resize((256, 256)),
#              transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
#              transforms.RandomHorizontalFlip(),
#              transforms.RandomVerticalFlip(),
#              #transforms.RandomCrop(224),
#              transforms.ToTensor(),
#              normalize,
#          ]))
#    
#weights = make_weights_for_balanced_classes(train_data)
#weights = torch.DoubleTensor(weights)  
#sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
#
#train_loader = DataLoader(train_data,
#                        batch_size = 30,
#                        sampler = sampler)
#
#for i, (images, target) in enumerate(train_loader):
#        # 将图片和标签转化为tensor         
#        a = np.array(images)
#        c = a[0]
#        b = np.array(target)
#        d = b[0]
