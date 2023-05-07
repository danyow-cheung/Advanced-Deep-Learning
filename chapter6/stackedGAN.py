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
