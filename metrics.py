import numpy as np
from keras.utils.np_utils import to_categorical
from keras import backend as K
def kappa(t,y,eps=1e-15):
	if y.ndim == 1:
		y = to_categorical(y,nb_classes=5)
	if t.ndim == 1:
		t = to_categorical(t,nb_classes=5)
	
	num_scored_items, num_ratings = y.shape
	ratings_mat = K.tile(K.arange(0, num_ratings)[:,None],
						(1, num_ratings))
	ratings_squared = (ratings_mat - ratings_mat.T) ** 2
	weights = ratings_squared / (num_ratings - 1) ** 2

	y_norm = y / (eps + y.sum(axis=1)[:,None])
	y = y_norm

	hist_rater_a = K.sum(y, axis=0)
	hist_rater_b = K.sum(t, axis=0)
	conf_mat = K.dot(y.T, t)

	nom = K.sum(weights * conf_mat)
	denom = K.sum(weights * K.dot(hist_rater_a[:,None], 
					hist_rater_b[None,:])
					/ num_scored_items)

	return 1 - nom / denom
