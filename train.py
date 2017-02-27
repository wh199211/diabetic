import os
import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from scipy.misc import imread
import pandas as p
from keras.utils import np_utils
from losses import kappalogclipped
from metrics import kappa
from model import model
from generator import train_oversample_gen,train_gen

default_transfo_params = {'rotation': True, 'rotation_range': (0, 360),
                           'contrast': True, 'contrast_range': (0.7, 1.3),
                           'brightness': True, 'brightness_range': (0.7, 1.3)    ,
                           'color': True, 'color_range': (0.7, 1.3),
                           'flip': True, 'flip_prob': 0.5,
                           'crop': True, 'crop_prob': 0.4,
                           'crop_w': 0.03, 'crop_h': 0.04,
                           'keep_aspect_ratio': False,
                           'resize_pad': False,
                           'zoom': True, 'zoom_prob': 0.5,
                           'zoom_range': (0.00, 0.05),
                           'paired_transfos': False,
                           'rotation_expand': False,
                           'crop_height': False,
                           'extra_width_crop': True,
                           'rotation_before_resize': False,
                           'crop_after_rotation': True}

train_dir = '/media/wanghao/VisualSearch/kaggledrtrain/train_ds2/'
label = '/home/wanghao/code/diabetic/data/trainLabels.csv'
data = p.read_csv(label)
_img_list = data.image.values + '.jpeg'
_img_level = data.level.values
output_size = 512
batch_size = 64
input_height , input_width = (output_size , output_size)
output_dim = 5
num_channels = 3
leakness = 0.5
data_augmentation = True
nb_epoch = 200

def data(imgpath=None,labelpath=None,part=False):
    path = imgpath
    label = p.read_csv(labelpath) 
    img_list = label.image.values
    
    lst = []
    for i in img_list:
	
        img = imread(os.path.join(path,i))
        img1 = np.array(img.reshape((3,img.shape[0],-1)))
        lst.append(img1)
    X = np.asarray(lst)
    y = label.level.values
    y = y.reshape((-1,1))
    Y = np_utils.to_categorical(y,5)
    return X,Y





model = model(leakness=leakness)
sgd = SGD(lr = 0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse' , optimizer=sgd , metrics=['accuracy'])


#if not data_augmentation:
#    print 'Not using data augmentation'
#    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=200, validation_data=(X_test,Y_test), 
#             shuffle=True)
#else:
#    print 'Using real-time data augmentation'
#    
#    train_gen = ImageDataGenerator(featurewise_center=False,
#    samplewise_center=False,
#    featurewise_std_normalization=False,
#    samplewise_std_normalization=False,
#    zca_whitening=False,
#    rescale=1./255,
#    rotation_range=30,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    shear_range=0.2,
#    horizontal_flip=True,
#    vertical_flip=True)
    
#    test_gen = ImageDataGenerator(rescale=1./255)
#    train_generator = train_gen.flow_from_directory('/home/wanghao/data/crop_train/',
#                                             target_size=(512, 512),
#                                            batch_size=32,classes=['0','1','2','3','4'],
 #                                           class_mode='categorical')
 #   test_generator = test_gen.flow_from_directory('/home/wanghao/data/test/crop_test/',
#                                         target_size=(512,512),
#                                        batch_size=32,classes=['0','1','2','3','4'],
#                                           class_mode='categorical')
    
#    model.fit_generator(train_generator,
#                       samples_per_epoch=35000,
#                       nb_epoch=200,
#                       validation_data=test_generator,
#                        nb_val_samples=800
#                       )
train_losses = []

for e in range(nb_epoch / 2):
	batches = 0

	for X_batch,Y_batch in train_oversample_gen(_img_list, _img_level,
												train_dir,
												default_transfo_params): 
		batches +=1

		batch_train_loss = model.train_on_batch(X_batch,Y_batch)
		train_losses.append(batch_train_loss)
		print 'Epoch',e , ': Batch', batches, 'loss = ',batch_train_loss
		 
		if batches > (len(_img_list) / batch_size):
			break

model.save('weights.h5')
