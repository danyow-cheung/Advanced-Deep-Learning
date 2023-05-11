from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from tensorflow_addons.layers import InstanceNormalization
import cifar10_utils
import mnist_svhn_utils
import other_utils
import datetime


# pip install tensorflow-addons 
def encoder_layer(
        inputs,
        filters=16,
        kernel_size=3,
        strides=2,
        activation='relu',
        instance_norm=True
    ):
    '''
    Build a generic encoder layer made of Conv2D-IN-LeakyReLu 
    IN is optional,LeakyReLU may be replaced by RELU
    '''
    conv = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization(axis=3)(x)
    if activation =='relu':
        x =  Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x 

def decoder_layer(
        inputs,
        paired_inputs,
        filters=16,
        kernel_size=3,
        strides=2,
        activation='relu',
        instance_norm=True
    ):
    '''
    Builds a generic decoder layer made of Conv2D-IN-LeakyReLU
    IN is optional,LeakyReLU may be replaced by RELU
    Arguments:(partial)
    
    '''
    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')


