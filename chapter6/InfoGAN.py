from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from gan import plot_images,generator,discriminator,train
from tensorflow.keras.utils import to_categorical
import tensorflow as tf 

import numpy as np
import math
import matplotlib.pyplot as plt
import os
s
def generator(inputs,image_size,activation='sigmoid',labels=None,codes=None):
    '''建立一個生成器模型
    BN-ReLU-Conv2DTranspose 堆疊去產生假圖片
    Arguments:
        inputs(Layer):      Input layer of the generator (the z-vector )
        image_size(int):    Target size of one side(assuming square image)
        activation(string): Name of output activation layer
        labels(tensor):     Input labels 
        codes(list):        2-dim disentangled codes for InfoGAN
    Returns:
        Model:              Generator Model
    '''
    image_resize = image_size//4 
    # 網路參數
    kernel_size = 5 
    layer_filters = [128,64,32,1]
    if labels is not None:
        if codes is None:
            # ACGAN labels 
            # concatenate z nosie vector and one-hot labels 
            inputs = [inputs,labels]
        else:
            # infoGAN codes
            # concatnate z noise vector 
            # one-hot lables and codes 1&2 
            x = np.concatenate(inputs,axis=1)
    elif codes is not None:
        # generator 0 of StackGAN 
        inputs = [inputs,codes]
    else:
        # default input is just 100-dim noise 
        x =inputs 

    x = Dense(image_resize*image_resize*layer_filters[0])(x)
    x = Reshape((image_resize,image_resize,layer_filters[0]))(x)
    for filters in layer_filters:
        # first two convolution layers use strides = 2 
        # the last two use strides =1 
        if filters>layer_filters[-2]:
            strides =2 
        else:
            strides = 1 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding='same')(x)

    if activation is not None:
        x = Activation(activation)(x)
    # genertor output is the synthesized image x 
    return Model(inputs,x,name='genrator')


def discriminator(inputs,activation='sigmoid',num_labels=None,num_codes=None):
    '''建立Discriminator摩西
    LeakyReLU-Conv2D 來從假圖片中鑑別真圖片
    
    Arguments:
        inputs(Layer):          Input layer of the discriminator (the image )
        activation(string):     Name of output activation layer
        num_labels(int):        Dimensions of one-hot labels for ACGAN&InfoGAN
        num_codes(int):         num_codes-dim Q network as output if StackedGAN or 2 Q networks if InfoGAN
        
    Returns:
        Model:                  Discriminator Model
        
    '''
    kernel_size = 5 
    layer_filters = [32,64,128,256]
    x = inputs 
    for filters in layer_filters:
        # first 3 conv use strides =2 
        # last strides =1 
        if filters == layer_filters[-1]:
            strides = 1 
        else:
            strides =2 
        x = LeaklyReLU(alpha=0.2)(x)
        x = Conv2D(
            filters=filters,kernel_size=kernel_size,
            strides=strides,padding='same',
        )(x)

        x = Flatten()(x)
        # default output is probability that the image is real 
        outputs = Dense(1)(x)
        if activation is not None:
            print(activation)
            outputs = Activation(activation)(outputs)
        if num_labels:
            # ACGAN and InfoGAN have 2nd output 
            # 2nd output is 10-dim one-hot vector of label
            layer = Dense(layer_filters[-2])(x)
            labels = Dense(num_labels)(layer)
            labels = Activation('softmax',name='label')(labels)

            if num_codes is None:
                outputs = [outputs,labels]

            else:
                # InfoGAN have 3rd and 4th outputs 
                # 3rd output is 1-dim continous Q of 1st c ginven x 
                code1 = Dense(1)(layer)
                code1 = Activation('sigmoid',name='code1')(code1)
                
                # 4th output is 1-dim continouts Q of 2nd c given x 
                code2 = Dense(1)(layer)
                code2 = Activation('sigmoid',name='code2')(code2)
                
        elif num_codes is not None:
            z0_recon = Dense(num_codes)(x)
            z0_recon = Activation('tanh',name='z0')(z0_recon)
            outputs = [outputs,z0_recon]

    return Model(inputs,outputs,name='discriminator')

# discriminator = gan.discriminator(inputs,num_labels=num_labels,num_codes=2)
# genrator = gam.generator(inputs,image_size,labels=labels,codes=[code1,code2])

def mi_loss(c,q_of_c_given_x):
    '''
    Mutal information ,assuming H(c) is constant 
    '''
    return K.mean(-K.sum(K.log(q_of_c_given_x+K.epsilon()) *c,axis=1))
def build_and_train_models(latent_size=100):
    '''Load the dataset ,build InfoGAN discriminator
    generator and adversrarial models 
    Call the InfoGAN train routine 
    '''
    (x_train,y_train),(_,_) = mnist.load_data()
    image_size = x_train.shape[1]
    # reshape data for CNN as (28,28,1) and normalize 
    x_trian = np.reshape(x_train,[-1,image_size,image_size,1])
    x_train = x_trian.astype("float32")/255 
    # train labels 
    num_labels = len(np.unique(y_train))
    y_trian = to_categorical(y_train)
    model_name = 'infogan_mnist'
    # 網路參數
    batch_size = 64 
    train_steps = 40000 
    lr =2e-4
    decay = 6e-8
    input_shape = (image_size,image_size,1)
    label_shape =(num_labels,)
    code_shape =(1,)

    # build discriminator model 
    inputs = Input(shape=input_shape,name='discriminator_input')

    # call discriminator builder with 4 outputs 
    # source,label and 2 codes 
    discriminator = discriminator(inputs,num_labels=num_labels,num_codes=2)
    # [1] uses Adam, but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr,decay=decay)
    loss = [
        'binary_crossentropy',
        'categorical_crossentropy',
        mi_loss,
        mi_loss
    ]

    loss_weights = [1.0,1.0,0.5,0.5]
    discriminator.compilie(
        loss=loss,
        loss_weights = loss_weights,
        optimizer = optimizer,
        metrics = ['accuracy']
    )
    discriminator.summary()

    # build generator model
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape,name='z_input')
    labels = Input(shape=label_shape,name='labels')

    code1 = Input(shape=code_shape,name='code1')
    code2 = Input(shape=code_shape,name='code2')
    # call genertor with inputs 
    # labels and codes as total inputs to generator 
    generator = generator(
        inputs,image_size,labels=labels,codes=[code1,code2]
    )
    generator.summary()
    # build adversarial model = generator + discriminator 
    optimizer = RMSprop(lr=lr*0.5,decay = decay*0.5)
    discriminator.trainable =False 

    inputs = [inputs,labels,code1,code2]
    adversarial = Model(inputs,discriminator(generator(inputs)),name=model_name)

    adversarial.compile(
        loss=loss,
        loss_weights = loss_weights,
        optimizer=optimizer,
        metrics = ['accuracy']
    )
    adversarial.summary()

    # 訓練
    models = (generator,discriminator,adversarial)
    data = (x_train,y_train)
    params = (
        batch_size,latent_size,train_steps,num_labels,model_name
    )

def train(models,data,params):
    '''
    Train the Discriminator and Adversarial networks
    交替地批量訓練鑑別器和對抗網絡。
    鑑別器首先用真實和虛假的圖像進行訓練，
    相應的one-hot標籤和連續代碼。
    接下來用假圖像假裝訓練對抗性
    是真實的，對應的one-hot labels和continuous codes。
    每個 save_interval 生成樣本圖像。
    Arguments
        models(Models):     Generator,Discriminator,Adversarial models 
        data(tuple):        x_train,y_train data 
        params(tuple):      Network parameters
    '''
    # the GAN models 
    generator,discriminator,adversarial = models

    # image and their one-hot labels 
    x_train,y_train = data 
    # network parameters 
    batch_size,latent_size,train_steps,num_labels ,model_name = params

    # the generator image is saved every 500  steps
    save_interval = 500 
    # noise vector to see how the genrator output 
    # envolves during training 
    noise_input = np.random.uniform(-1.0,1.0,size=[16,latent_size])

    # random class labels and codes 
    noise_label = np.eye(num_labels)[np.arange(0,16) % num_labels]
    noise_code1 = np.random.normal(scale=0.5,size=[16,1])
    noise_code2 = np.random.normal(scale=0.5,size=[16,1])

    # number of elements in train dataset 
    train_size = x_train.shape[0]
    # 2023.5.6
