from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,MaxoutDense, Dropout, Flatten,Merge, Reshape,Activation
from keras.layers.advanced_activations import LeakyReLU

def KerasNet(leakness = 0.5):

	inputs = Input(shape=(3,512,512))
	x = Convolution2D(32,7,7,init='orthogonal', subsample=(2,2), border_mode='same')(inputs)
	x = LeakyReLU(alpha=leakness)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
	x = Convolution2D(32,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

	x = Convolution2D(64,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(64,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

	x = Convolution2D(128,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(128,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(128,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(128,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

	x = Convolution2D(256,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(256,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(256,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = Convolution2D(256,3,3, init='orthogonal',subsample=(1,1), border_mode='same')(x)
	x = LeakyReLU(alpha=leakness)(x)
	x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

	x = Dropout(0.5)(x)
	x = Flatten()(x)

	x = MaxoutDense(512, nb_feature=4,init='orthogonal')(x)
	x = Dropout(0.5)(x)
	x = MaxoutDense(512, nb_feature=4,init='orthogonal')(x)
	x = Dense(5,init='orthogonal')(x)
	x = Activation('softmax')(x)

	model = Model(inputs, x)

	return model
