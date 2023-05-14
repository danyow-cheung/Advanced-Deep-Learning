from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

# reparameterization trick 
# instead of sampling from Q(z|X), sample eps =N(0,I)
# z = z_mean + sqrt(var)*eps 
def sampling(args):
    '''
    Reparameterization trick by sampling fr an isotropic unit Gaussian
    # Arguments:
        args(tensor):   mean and log of variance of Q(z|X)
    # Returns:
        z(tensor):      sampled latent vector 
    '''
    z_mean,z_log_var = args 
    # K is the keras backend
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default random_normal has meam = 0 and std=1.0 
    epsilon = K.random_normal(shape=(batch,dim))
    return z_mean+K.exp(0.5*z_log_var)*epsilon

# MNIST dataset 
(x_train,y_train),(x_test,y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size*image_size
x_train = np.reshape(x_train,[-1,original_dim])
x_test = np.reshape(x_test,[-1,original_dim])
x_train = x_train.astype('float32')/255 
x_test = x_test.astype('float32')/255

# network parameters 
input_shape = (original_dim,)
intermediate_dim = 512 
batch_size = 128 
latent_dim = 2 
epochs = 50 

# VAE model = encoder + decoder 
# build encoder model 
inputs = Input(shape=input_shape,name='encoder_input')
x = Dense(intermediate_dim,activation='relu')(inputs)
z_mean = Dense(latent_dim,name='z_mean')(x)
z_log_var = Dense(latent_dim,name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input 
# with the tensorflow backend 
z = Lambda(sampling,output_shape=(latent_dim,),name='z')([z_mean,z_log_var])
# instantiate encoder model 
encoder = Model(inputs,[z_mean,z_log_var,z],name='encoder')
# build the decoder model 
latent_inputs = Input(shape=(latent_dim,),name='z_sampling')
x = Dense(intermediate_dim,activation='relu')([latent_inputs])
outputs = Dense(original_dim,activation='sigmoid')(x)
# instantiate decoder model 
decoder = Model(latent_inputs,outputs,name='decoder')
# instantiate VAE model 
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs,outputs,name='vae_mlp')
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    help_ = "Load tf model trained weights "
    parser.add_argument('-w','--weights',help=help_)
    help_ = 'Use binary cross entropy instead of mse(default)'
    parser.add_argument('--bce',help=help_,action='store_true')
    args = parser.parse_args()
    models = (encoder,decoder)

    data=(x_test,y_test)
    # VAE loss = mse_loss or xent_loss + kl_loss 
    if args.bce:
        reconstruction_loss = binary_crossentropy(inputs,outputs)
    else:
        reconstruction_loss = mse(inputs,outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1+z_log_var-K.square(z_mean)-K.exp(z_log_var)
    kl_loss = K.sum(kl_loss,axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss+kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    