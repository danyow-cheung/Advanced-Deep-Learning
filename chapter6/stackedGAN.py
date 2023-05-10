from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
from lib import gan 

def build_encoder(inputs,num_labels=10,feature1_dim=256):
    '''
    build the classifier (Encoder) Model sub networks
    Two sub networks:
    1. Encoder0: Image to feature1 (intermediate latent feature)
    2. Encoder1: feature1 to labels
    
    # Arguments:
        inputs(Layers):     x - images ,featues1 - feature1 layer output
        num_labels(int):    number of class label 
        feature_dim(int):   feature1 dimensionality
    # Returns:
        enc0,enc1(Models):  Description below 
    '''
    kernel_size = 3 
    filters = 64 

    x,feature1 = inputs 
    y = Conv2D(filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu')(x)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(y)
    
    y = MaxPooling2D()(y)

    y = Flatten()(y)
    feature1_output = Dense(feature1_dim,activation='relu')(y)
    enc0 = Model(inputs=x,outputs=feature1_output,name='encoder0')

    # encoder1 or enc1 
    y = Dense(num_labels)(feature1)
    labels = Activation('softmax')(y)
    enc1 = Model(inputs=feature1,outputs=labels,name='encoder1')
    return enc0,enc1 

def build_generator(latent_code,latent_size,image_size,feature1_dim=256):
    '''
    Build Generator Model sub networks 
    Two sub networks 1)Class and noise to feature1 (intermediate feature)
                     2) feature to image 
    
    # Arguments
        latent_codes(Layer): discreter code (labels)
            noise and feature1 features 
        image_size(int):     Target size of one side(assuming square image)
        feature1_dim(int):   features1 dimensionality
        
        # Returns 
            gen0,gen1(Models): Description below 
    '''
    # latent codes and network parameters 
    labels,z0,z1 ,feature1 = latent_code 

    # gan inputs 
    inputs = [labels,z1]
    x = concatenate(inputs,axis=1)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(512,activatio='relu')(x)
    x = BatchNormalization()(x)
    take_feature1 = Dense(feature1_dim,activation='relu')(x)
    # gen1:Classes and noise (feature2 + z1) to featue1 
    fake_feature1 = Dense(feature1_dim,activation='relu')(x)
    gen1 = Model(inputs,fake_feature1,name='gen1')
    gen0 = gan.ganerator(feature1,image_size,codes=20)
    return gen0,gen1



def build_Discriminator(inputs,z_dim=50):
    '''
    Build Discriminator 1 Model 
    Classification feature1 (features) as real/fake image and recovers 
    the input noise or latent code (by minimizing entropy loss)
    # Arguments
        z_dim(int):     noise dimensionality
    # Returns 
        dis1(Model):    feature1 as real/fake and recovered latent code
    '''
    # inputs is 256-dim feature1 
    x = Dense(256,activation='relu')(inputs)
    x = Dense(256,activation='relu')(x)

    # first output is probability that feature1 is real 
    f1_sources = Dense(1)(x)
    f1_sources = Activation('sigmoid',name='feature1_source')(f1_sources)

    # z1 reconstructin (Q1 network)
    z1_recon = Dense(z_dim)(x)
    z1_recon = Activation('tanh',name='z1')(z1_recon)
    discriminator_outputs = [f1_sources,z1_recon]
    dis1 = Model(inputs,discriminator_outputs,name='dis1')
    return dis1 

def build_and_train_models():
    ''' Load the dataset,build Stacked GAN discriminator,generator,and adversarial models
    Call the stacked GAN train routine'''
    # load the dataset 
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    # reshape and normalize images 
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train,[-1,image_size,image_size,1])
    x_train = x_train.astype('float32')/255 
    x_test = np.reshape(x_test,[-1,image_size,image_size,1])
    x_test = x_test.astype('float32')/255 

    # number of labels 
    num_labels = len(np.unique(y_train))
    # to one-hot vector 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model_name = 'stackedgan_mnist'
    # 网络参数
    batch_size = 64 
    train_steps = 10000 
    lr = 2e-4 
    decay = 6e-8
    input_shape = (image_size,image_size,1)
    label_shape = (num_labels,)
    z_dim = 50 
    z_shape = (z_dim,)
    feature1_dim = 256 
    feature1_shape = (feature1_dim,)

    # build discriminator 0 and q network 0 models 
    inputs = Input(shape=input_shape,name='discriminator0_input')
    dis0 = gan.discriminator(inputs,num_codes=z_dim)

    optimizers = RMSprop(lr=lr,decay=decay)
    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0,10.0]
    dis0.compilie(loss=loss,loss_weights=loss_weights,
                  optimizers=optimizers,
                  metrics=['accuracy'])
    
    dis0.summary()

    # 
