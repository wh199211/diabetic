import os
import numpy as np
from scipy.misc import imread
import pandas as p
from keras.utils import np_utils

output_size = 512
batch_size = 64
input_height , input_width = (output_size , output_size)
output_dim = 5
num_channels = 3
leakness = 0.5
data_augmentation = True

def data(imgpath=None,labelpath=None,part=False,m=None):
    path = imgpath
    label = p.read_csv(labelpath) 
    img_list = label.image.values
    img_list += '.jpeg'
    if part:
        img_list = img_list[:2000]
    lst = []
    for i in img_list:
        img = imread(os.path.join(path,i))
        img1 = np.array(img.reshape((3,img.shape[0],-1)))
        print i
        lst.append(img1)
    X = np.asarray(lst)
    np.save('%s.npy'%m,X)
    del X
    y = label.level.values
    y = y.reshape((-1,1))
    Y = np_utils.to_categorical(y,5)
    np.save('%s_label'%m,Y)
    del Y

data(imgpath='/media/wanghao/VisualSearch/kaggledrtrain/train_resize/'
                       ,labelpath='/home/wanghao/kaggle/kaggle_diabetic_retinopathy/data/trainLabels.csv',m='train')
data(imgpath='/media/wanghao/VisualSearch/kaggledrtest/test_resize/'
                    ,labelpath='/home/wanghao/kaggle/kaggle_diabetic_retinopathy/data/retinopathy_solution.csv'
                    ,part=True,m='test')
