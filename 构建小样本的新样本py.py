import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

datagen = ImageDataGenerator(
#    width_shift_range = 0.1,
#    height_shift_range = 0.1,
#    zoom_range = 0.1,
#    rotation_range = 90,
    horizontal_flip = True,
    vertical_flip = True,

    )

label2num = {
        'OCEAN':0,
        'MOUNTAIN':1,
        'LAKE':2,
        'FARMLAND':3,
        'DESERT':4,
        'CITY':5
    }
train_path = './train/'   
label_csv = pd.read_csv('train_label.csv', header = None, encoding='utf-8')
label_csv[2] = label_csv[1].map(lambda x: label2num[x])
label_csv[0] = label_csv[0].map(lambda x: train_path + x)
label_csv = label_csv.drop( [1],  axis=1)

for index, row in label_csv.iterrows():
    if row[2] == 3:
        x = image.load_img( row[0] )
        x = np.array(x)
        x = x[np.newaxis,:,:,:]
        datagen.fit(x)
        num = 30
        i = 1
        if not os.path.exists('./%s/' % row[2]):
            os.makedirs('./%s/' % row[2])
        for x_batch in datagen.flow( x, batch_size = 2,
                                     save_to_dir = './%s/' % row[2],
                                     save_prefix = num,
                                     save_format = 'jpg'):
            num = num+1
            i += 1
            if i > 1:
                break
    if row[2] == 2:
        x = image.load_img( row[0] )
        x = np.array(x)
        x = x[np.newaxis,:,:,:]
        datagen.fit(x)
        num = 20
        i = 1
        if not os.path.exists('./%s/' % row[2]):
            os.makedirs('./%s/' % row[2])
        for x_batch in datagen.flow( x, batch_size = 2,
                                     save_to_dir = './%s/' % row[2],
                                     save_prefix = num,
                                     save_format = 'jpg'):
            num = num+1
            i += 1
            if i > 2:
                break
    if row[2] == 5:
        x = image.load_img( row[0] )
        x = np.array(x)
        x = x[np.newaxis,:,:,:]
        datagen.fit(x)
        num = 50
        i = 1
        if not os.path.exists('./%s/' % row[2]):
            os.makedirs('./%s/' % row[2])
        for x_batch in datagen.flow( x, batch_size = 2,
                                     save_to_dir = './%s/' % row[2],
                                     save_prefix = num,
                                     save_format = 'jpg'):
            num = num+1
            i += 1
            if i > 8:
                break