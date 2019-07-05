# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:05:41 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import random
import torch.nn.functional as F
import os
import time

class_num =6
test_size = 0.9
val_ratio = 0.1
border_name = "same"
l2_lr = 0.01
batch_size=10
num_epochs = 1
use_cuda =False
train_path = '/content/drive/My Drive/tiangong/train/'
workers = 0
input_size = 256* 256
num_classes = 16
num_epochs = 10
batch_size = 30
learning_rate = 1e-3



evaluate = False
stage_epochs = [20, 10, 10]  
# 初始学习率
lr = 1e-4
# 学习率衰减系数 (new_lr = lr / lr_decay)
lr_decay = 5
# 正则化系数
weight_decay = 1e-4

# 参数初始化
stage = 0
start_epoch = 0
total_epochs = sum(stage_epochs)
best_precision = 0
lowest_loss = 100
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # (3,23,23) -->(128,21,21)
        self.conv1 = torch.nn.Conv2d(3, 128, 7, stride=3)
        # (128,11,11) -->(100,9,9)
        self.conv2 = torch.nn.Conv2d(128, 100, 7, stride=3)
        
        self.conv3 = torch.nn.Conv2d(128, 100, 7, stride=3)
        # (100,5,5)->(200)
        self.fcl1 = torch.nn.Linear(100 * 7* 7, 200, bias=True)
        # (200)->(84)
        self.fcl2 = torch.nn.Linear(200, 84, bias=True)
        # (84->16)
        self.fcl3 = torch.nn.Linear(84, 6, bias=True)

        # Parameters:
        # x would be the input tensor to be forward propgated

    def forward(self, x):

        # 输入x经过卷积conv1之后, 经过激活函数ReLU(原来这个词是激活函数的意思),使用2x2的窗口进行最大池化Maxpooling,然后更新到x.
        # (128,21,21)->(128,11,11)
        x = F.relu(F.max_pool2d(self.conv1(x), 2, padding=1))
        # (100,9,9)->(100,5,5)
        x = F.relu(F.max_pool2d(self.conv2(x), 2, padding=1))
        x = x.view(-1, 100 * 7 * 7)
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = F.log_softmax(self.fcl3(x), dim=1)  # dim = dimension along which log softmax will be computed
        return x

    def name(self):
        return "CNN"

def default_loader(path):
#    print(path[0])
#    a = Image.open(path[0]).convert('RGB')
#    print(a)
    return Image.open(path).convert('RGB')

# 训练集图片读取
class TrainDataset(Dataset):
    '''
    Dataset 是一个抽象的类,自定义的Dataset需要继承Dataset这个类,同时重写__getitem__()
    和__len__() 两个方法

    '''

    def __init__(self, data_csv, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in data_csv.iterrows():
            imgs.append((row[0], row[2]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    

    def __getitem__(self,index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)



class ValDataset(Dataset):
    '''
    Dataset 是一个抽象的类,自定义的Dataset需要继承Dataset这个类,同时重写__getitem__()
    和__len__() 两个方法

    '''

    def __init__(self, data_csv, transform=None, target_transform=None, loader=default_loader):
        pic_list = []
        for pic in (data_csv[0].values):
            pic_list.append(train_path + pic) 
        self.picname = pic_list
        self.labels = list(data_csv[2])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    

    def __getitem__(self,index):
        img = self.loader(self.picname)
        if self.transform is not None:
            img = self.transform(img)
        labels = self.labels
        return img, labels
    

    def __len__(self):
        return len(self.labels)


def calculate_acc(cnn, test_loader):
    total_count = 0
    correct_count = 0
    criterion = nn.CrossEntropyLoss()
    if (use_cuda):
        cnn = cnn.cuda()
    else:
        cnn = cnn.cpu()

    for batch_idx, (x, target) in enumerate(test_loader):
        if (use_cuda):
            x, target = x.cuda(), target.cuda()
        else:
            x, target = Variable(x).double(), Variable(target)

        output = cnn(x)
        #loss = criterion(output, target-1)
        _, pred_label = torch.max(output.data, 1)      
        # total_count 最后是所有训练或者测试数据的总和
        total_count += x.data.size()[0]
        correct_count += ((pred_label+1 == target.data).sum()).item()
        acc = (correct_count * 1.0) / total_count
    return acc



def train(cnn,train_loader,test_loader,batch_size):

    if (use_cuda):
        cnn = cnn.cuda()  # GPU model
    else:
        cnn = cnn.cpu()
        cnn = cnn.double()

    for epoch in range(num_epochs):
        for batch_idx, (x, target) in enumerate(train_loader):
            # 1.得到数据并包装数据
            if(use_cuda):
                x, target = x.cuda(), target.cuda()
            else:
                x, target = Variable(x).double(), Variable(target)
            # 2.梯度清零
            optimizer.zero_grad()
            # 3.前向传播 + 反向传播 + 优化
            output = cnn(x)
            # 此时的 loss 是每一个batch_size中的损失函数
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if((epoch+1)%1== 0):
            # 此时打印的loss是每次训练中最后一个batch的损失函数
            print ('epoch: %d, train loss: %.6f' %(epoch, loss))
        train_acc = calculate_acc(cnn,train_loader)
        if epoch ==0:
            last_acc = train_acc
        test_acc = calculate_acc(cnn,test_loader)
        print('train accurate: %.5f'%(train_acc))
        print('val accurate: %.5f'%(test_acc))
        if last_acc<=train_acc:
            # 只保存神经网络的模型参数 
            torch.save(cnn.state_dict(),"/content/drive/My Drive/tiangong//model/cnn%d_train%.4f_test%.4f.pkl"%(epoch,train_acc,test_acc))
            print("save networks for epoch:",epoch)
            last_acc = train_acc

# 数据增强：在给定角度中随机进行旋转
class FixedRotation(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        return fixed_rotate(img, self.angles)

def fixed_rotate(img, angles):
    angles = list(angles)
    angles_num = len(angles)
    index = random.randint(0, angles_num - 1)
    return img.rotate(angles[index])

if __name__ == '__main__':
    #### 1.加载并规范化数据集
    label2num= {
        'OCEAN':0,
        'MOUNTAIN':1,
        'LAKE':2,
        'FARMLAND':3,
        'DESERT':4,
        'CITY':5
    }
    file_name = 'tg'
    label_csv = pd.read_csv('/content/drive/My Drive/tiangong/train_label.csv',header = None,encoding='utf-8')
    ori_label = label_csv[1]
    label_csv[2] = label_csv[1].map(lambda x:label2num[x])
    train_csv, val_csv = train_test_split(label_csv, test_size=val_ratio, random_state=666, stratify=label_csv[2])
    
    
    train_csv[0] = train_csv[0].map(lambda x:train_path+x)
    val_csv[0] = val_csv[0].map(lambda x:train_path+x)
    train_csv = train_csv.drop([1], axis=1)
    val_csv = val_csv.drop([1], axis=1)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_data = TrainDataset(train_csv,
                                  transform=transforms.Compose([
                                      transforms.Resize((256, 256)),
    #                                  transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.RandomGrayscale(),
                                      # transforms.RandomRotation(20),
                                      FixedRotation([0, 90, 180, 270]),
                                     # transforms.RandomCrop(384),
                                      transforms.ToTensor(),
                                      normalize,
                                  ]))
        
        
        
    val_data = TrainDataset(val_csv,
                              transform=transforms.Compose([
                                  transforms.Resize((256, 256)),
                                 # transforms.CenterCrop(384),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))
    #    
        
    #test_data = TestDataset(test_data_list,
    #                        transform=transforms.Compose([
    #                           # transforms.Resize((400, 400)),
    #                           # transforms.CenterCrop(384),
    #                            transforms.ToTensor(),
    #                            normalize,
    #                        ]))
       
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    
    #val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
    #test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
    #X_train_nei, X_test_nei, Y_train, Y_test, idx_train, idx_test = load_HSI_data(n_components)
    #X_train_nei = np.transpose(X_train_nei, (0, 3, 1, 2))
    #X_test_nei = np.transpose(X_test_nei, (0, 3, 1, 2))
    #train_data = TrainDataset(X_train_nei, Y_train)
    #test_data = TrainDataset(X_test_nei, Y_test)
    ## 生成图片迭代器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    ####### 2. 定义一个卷积神经网络
    cnn = CNN()
    model = cnn.double()
    #
    #
    ####### 3.定义一个损失函数和优化器
    # 使用交叉熵损失函数
    if not os.path.exists('/content/drive/My Drive/tiangong/model/%s' % file_name):
        os.makedirs('/content/drive/My Drive/tiangong/model/%s' % file_name)
    if not os.path.exists('/content/drive/My Drive/tiangong/result/%s' % file_name):
        os.makedirs('/content/drive/My Drive/tiangong/result/%s' % file_name)
    # 创建日志文件
    if not os.path.exists('/content/drive/My Drive/tiangong/result/%s.txt' % file_name):
        with open('/content/drive/My Drive/tiangong/result/%s.txt' % file_name, 'w') as acc_file:
            pass
    with open('/content/drive/My Drive/tiangong/result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))
    criterion = nn.CrossEntropyLoss().cuda()
    
    # 优化器，使用带amsgrad的Adam
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
    
    if evaluate:
        validate(val_loader, model, criterion)
    else:
        # 开始训练
        for epoch in range(start_epoch, total_epochs):
            # train for one epoch
            train(cnn, train_loader, val_loader, batch_size = batch_size)
            # evaluate on validation set
            precision, avg_loss = validate(val_loader, model, criterion)
    
            # 在日志文件中记录每个epoch的精度和loss
            with open('/content/drive/My Drive/tiangong/result/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))
    
            # 记录最高精度与最低loss，保存最新模型与最佳模型
            is_best = precision > best_precision
            is_lowest_loss = avg_loss < lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
                'stage': stage,
                'lr': lr,
            }
            save_checkpoint(state, is_best, is_lowest_loss)
    
            # 判断是否进行下一个stage
            if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar' % file_name)['state_dict'])
                print('Step into next stage')
                with open('/content/drive/My Drive/guangdong/result/%s.txt' % file_name, 'a') as acc_file:
                    acc_file.write('---------------Step into next stage----------------\n')
    
    # 记录线下最佳分数
    with open('/content/drive/My Drive/tiangong/result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('* best acc: %.8f  %s\n' % (best_precision, os.path.basename(__file__)))
    with open('/content/drive/My Drive/tiangong/result/best_acc.txt', 'a') as acc_file:
        acc_file.write('%s  * best acc: %.8f  %s\n' % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision, os.path.basename(__file__)))
    
    # 读取最佳模型，预测测试集，并生成可直接提交的结果文件
    best_model = torch.load('/content/drive/My Drive/tiangong/model/%s/model_best.pth.tar' % file_name)
    model.load_state_dict(best_model['state_dict'])
    test(test_loader=test_loader, model=model)
    
    # 释放GPU缓存
    torch.cuda.empty_cache()