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




image_size = 180 
# network parameters 
input_shape = (image_size,image_size,1)
batch_size = 128 
kernel_size = 3 
filters = 16 
latent_dim = 2 
epochs = 30 

# VAE model = encoder + decoder 
# build encoder model 
inputs = Input(shape=input_shape,name='encoder_input')
x = inputs 
for i in range(2):
    filter *= 2 
    x = Conv2D(filters=filters,kernel_size=kernel_size,activation='relu',strides=2,padding='same')(x)

# shape info needed to build decoder model 
shape =K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16,activation='relu')(x)

z_mean = Dense(latent_dim,name='z_mean')(x)
# 20230515