from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Flatten, Lambda

from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import concatenate
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os



(x_train,y_train),(x_test,y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size*image_size
x_train = np.reshape(x_train,[-1,original_dim])
x_test = np.reshape(x_test,[-1,original_dim])
x_train = x_train.astype('float32')/255 
x_test = x_test.astype('float32')/255


# compute the number of labels 
num_labels = len(np.unique(y_train))



image_size = 180 
# network parameters 
input_shape = (image_size,image_size,1)
label_shape = (num_labels,)
batch_size = 128 
kernel_size = 3 
filters = 16 
latent_dim = 2 
epochs = 30 

# VAE model = encoder + decoder 
# build encoder model 
inputs = Input(shape=input_shape,name='encoder_input')
y_lables = Input(shape=label_shape,name='class_labels')

x = Dense(image_size*image_size)(y_lables)
x = Reshape((image_size,image_size,1))(x)
x = concatenate([inputs,x])

for i in range(2):
    filter *= 2 
    x = Conv2D(filters=filters,kernel_size=kernel_size,activation='relu',strides=2,padding='same')(x)

# shape info needed to build decoder model 
shape =K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16,activation='relu')(x)

z_mean = Dense(latent_dim,name='z_mean')(x)
z_log_var = Dense(latent_dim,name='z_log_var')(x)

# use reparameterization trick to push the samping out as input 
z = Lambda(samping,output_shape=(latent_dim,),name='z')([z_mean,z_log_var])

# instantiate encoder model 
encoder = Model([inputs,y_lables],[z_mean,z_log_var,z],name='encoder')

# build decoder model 
latent_inputs = Input(shape=(latent_dim,),name='z_sampling')
x = concatenate([latent_inputs,y_lables])
x = Dense(shape[1]*shape[2]*shape[3],activation='relu')(x)
x = Reshape((shape[1],shape[2],shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,kernel_size=kernel_size,activation='relu',strides=2,padding='same')(x)
    filters //=2 
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          name='decoder_output')(x)

# instantiate decoder model 
decoder = Model([latent_inputs,y_lables],outputs,name='decoder')
# instantiate vae model 
outputs = decoder([encoder([inputs,y_lables])[2],y_lables])
cvae = Model([inputs,y_lables],outputs,name='cvae')
