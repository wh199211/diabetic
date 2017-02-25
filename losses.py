from keras import backend as K
import numpy as np
from keras.objectives import categorical_crossentropy as cross_entropy

def log_loss(t,y):
	#TODO

	return

def kappa_loss(t,y , eps = 1e-15):
	#TODO

	num_scored_items = y.shape[0]
	num_ratings = 5
	tmp = K.tile(K.arange(0, num_ratings).reshape((num_ratings, 1)),
				(1,num_ratings))
	#tmp = K.cast_to_floatx(tmp)

	weights = (tmp - tmp.T) ** 2 / (num_ratings - 1) ** 2

	y_norm = y / (eps + y.sum(axis=1).reshape((num_scored_items, 1)))

	hist_rater_a = y_norm.sum(axis=0)
	hist_rater_b = t.sum(axis=0)

	conf_mat = K.dot(y_norm.T , t)

	nom = K.sum(weights * conf_mat)
	denom = K.sum(weights * K.dot(hist_rater_a.reshape((num_ratings , 1)),
								hist_rater_b.reshape((1, num_ratings))) / 
					num_scored_items)
	return - (1 - nom / denom)

def kappalogclipped(t,y,log_scale=0.5,log_cutoff=0.9):
	#TODO
	return kappa_loss(t,y) + log_scale * \
			K.clip(cross_entropy(y,t),log_cutoff ,10 ** 3 )
