import os
import numpy as np
import theano.tensor as T
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,MaxPooling2D,MaxoutDense, Dropout, Flatten,Merge, Reshape,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from scipy.misc import imread
import pandas as p
from keras.utils import np_utils

model = Sequential()

output_size = 512
batch_size = 64
input_height , input_width = (output_size , output_size)
output_dim = 5
num_channels = 3
leakness = 0.5
data_augmentation = True

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





model = Sequential()
model.add(Convolution2D(32,7,7, subsample=(2,2), border_mode='same',input_shape=(3,512,512)))
model.add(LeakyReLU(alpha=leakness))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Convolution2D(32,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Convolution2D(128,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(128,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(128,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(128,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Convolution2D(256,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(256,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(256,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(Convolution2D(256,3,3, subsample=(1,1), border_mode='same'))
model.add(LeakyReLU(alpha=leakness))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(MaxoutDense(512, nb_feature=4))
model.add(Dropout(0.5))
model.add(MaxoutDense(512, nb_feature=4))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='mse' , optimizer='rmsprop' , metrics=['accuracy'])


if not data_augmentation:
    print 'Not using data augmentation'
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=200, validation_data=(X_test,Y_test), 
             shuffle=True)
else:
    print 'Using real-time data augmentation'
    
    train_gen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)
    
    test_gen = ImageDataGenerator(rescale=1./255)
    train_generator = train_gen.flow_from_directory('',
                                             target_size=(512, 512),
                                            batch_size=32,classes=['0','1','2','3','4'],
                                            class_mode='categorical')
    test_generator = test_gen.flow_from_directory('/media/b/DC0C9F700C9F4504/crop_test/',
                                           target_size=(512,512),
                                           batch_size=32,classes=['0','1','2','3','4'],
                                           class_mode='categorical')
    
    model.fit_generator(train_generator,
                       samples_per_epoch=35000,
                       nb_epoch=200,
                       validation_data=test_generator,
                        nb_val_samples=800
                       )