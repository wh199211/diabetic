from scipy.misc import imread
import os
import pandas as p
import random
from keras.utils.np_utils import to_categorical
import numpy as np
from augmentation import load_image_and_process,make_thumb

label = '/home/wanghao/code/diabetic/data/trainLabels.csv'
sample_coefs = [0,7,3,22,25]

data = p.read_csv(label)

_img_list = data.image.values
_img_level = data.level.values

def sample_dr(level=0,img_list=_img_list,img_level=_img_level):
	l = []
	for i,j in enumerate(img_list):
		if img_level[i] == level:
			l.append(j)
	return l

def train_oversample_gen(img_list,img_level,
						img_dir, params, 
						 nb_epoch,batch_size=64):
	if nb_epoch < 100:
		#TODO
		train_0 = sample_dr(level=0,img_list = img_list,img_level=img_level)
		train_1 = sample_dr(level=1,img_list = img_list,img_level=img_level)
		train_2 = sample_dr(level=2,img_list = img_list,img_level=img_level)
		train_3 = sample_dr(level=3,img_list = img_list,img_level=img_level)
		train_4 = sample_dr(level=4,img_list = img_list,img_level=img_level)

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

		while 1:
			train_uniform = random.sample(train_0,12) + \
				random.sample(train_1,13) + \
				random.sample(train_2,13) + \
				random.sample(train_3,13) + \
				random.sample(train_4,13)

			y_uniform = [0] * 12 + [1] * 13 + [2] * 13 + [3] * 13 + [4] * 13

			m = range(batch_size)
			random.shuffle(m)
			train = []
			y = []
			for i in m:
				train.append(train_uniform[i])
				y.append(y_uniform[i])
			Y = to_categorical(y,nb_classes=5)
			x_list = []
			for i in train:
			#	print i
				img = load_image_and_process(i,prefix_path=img_dir,
											transfo_params=params)
			#	theano
				img_arr = np.array(img.reshape((3,img.shape[0],-1)))
			#	tensorflow
			#	img_arr = np.array(img)
				x_list.append(img_arr)		
			X = np.asarray(x_list)

			yield (X, Y)

	else:
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
def valid_gen(img_list, img_level, img_dir, 
				params, batch_size=64):
	valid_img = img_list[:640]
	valid_level = img_level[:640]
	
	x_list = []
	for i in valid_img:
		img = load_image_and_process(i,prefix_path=img_dir,
									transfo_params=params)
		img_arr = np.array(img.reshape((3,img.shape[0],-1)))
		x_list.append(img_arr)
	
	X = np.asarray(x_list)
	Y = to_categorical(valid_level, nb_classes=5)

	return (X,Y)
def test_gen(img_list, img_dir,batch_size=64):
	#TODO
	x_list = []
	for i in img_list:
		img = imread(os.path.join(img_dir,i))
		im_resize = make_thumb(img)
		img_arr = np.array(im_resize.reshape((3,im_resize.shape[0],-1)))
		x_list.append(img_arr)
		
	X = np.asarray(x_list)
	return X 
