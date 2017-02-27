from scipy.misc import imread
import os
import pandas as p
import random
from keras.utils.np_utils import to_categorical
import numpy as np
from augmentation import load_image_and_process

label = '/home/wanghao/code/diabetic/data/trainLabels.csv'
sample_coefs = [0,7,3,22,25]

data = p.read_csv(label)

_img_list = data.image.values
_img_level = data.level.values

def sample_dr(level=0):
	l = []
	for i,j in enumerate(img_list):
		if img_level[i] == level:
			l.append(j)
	return l

def train_oversample_gen(img_list,img_level,img_dir, params,
						batch_size=64):
	#TODO
	train_0 = sample_dr(level=0)
	train_1 = sample_dr(level=1)
	train_2 = sample_dr(level=2)
	train_3 = sample_dr(level=3)
	train_4 = sample_dr(level=4)

	#train_list = img_list
	#train_list += img_list[coefs[1] * train_1]
	#train_list += img_list[coefs[2] * train_2]
	#train_list += img_list[coefs[3] * train_3]
	#train_list += img_list[coefs[4] * train_4]
	
	#label_list = img_level
	#label_list.extend([1] * coefs[1])
	#label_list.extend([2] * coefs[2])
	#label_list.extend([3] * coefs[3])
	#label_list.extend([4] * coefs[4])

	Y = [0] * 12 + [1] * 13 + [2] * 13 + [3] * 13 + [4] * 13
	Y = to_categorical(Y,nb_classes=5)
	while 1:
		train = random.sample(train_0,12) + 
			random.sample(train_1,13) +
			random.sample(train_2,13) +
			random.sample(train_3,13) +
			random.sample(train_4,13)
		train += '.jpeg'
		x_list = []
		for i in train:
			img = load_image_and_process(i,prefix_path=img_dir,
										transfo_params=params)
			img_arr = np.array(img.reshape((3,img.shape[0],-1)))
			x_list.append(img_arr)		
		X = np.asarray(x_list)

		yield (X, Y)

def train_gen(img_list, img_level,img_dir,
				params,batch_size=64):
	img_list += 'jpeg'
	while 1:
		for i in range(len(img_list) / batch_size):
			batch_list = img_list[batch_size * i : batch_size * (i + 1)]
			y_list = img_level[batch_size * i : batch_size * (i + 1)]
			
			x_list = []
			for j in batch_list:
				img = load_image_and_process(j,prefix_path=img_dir,
											transfo_params=params)
				img_arr = np.array(img.reshape((3,img.shape[0],-1)))
				x_list.append(img_arr)
			X = np.asarray(x_list)
			Y = to_categorical(y_list,nb_classes=5)

			yield (X,Y)
	
def test_gen(self):
	#TODO
	return
