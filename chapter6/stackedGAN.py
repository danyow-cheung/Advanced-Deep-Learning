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

    optimizer = RMSprop(lr=lr,decay=decay)
    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0,10.0]
    dis0.compilie(loss=loss,loss_weights=loss_weights,
                  optimizers=optimizer,
                  metrics=['accuracy'])
    
    dis0.summary()

    # build discriminator 1 and 1 network 1 models
    input_shape = (feature1_dim, )
    inputs = Input(shape=input_shape, name='discriminator1_input')
    dis1 = build_Discriminator(inputs, z_dim=z_dim )
    # loss fuctions: 1) probability feature1 is real
    # (adversarial1 loss)
    # 2) MSE z1 recon loss (Q1 network loss or entropy1 loss)
    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 1.0]
    dis1.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    dis1.summary() # feature1 discriminator, z1 estimator

    # build generator models 
    feature1 = Input(shape=feature1_shape,name='feature1_input')
    labels = Input(shape=label_shape,name='labels')
    z1 = Input(shape=z_shape,name='z1_input')
    z0 = Input(shape=z_shape,name='z0_input')
    gen0,gen1 = build_generator(latent_codes,image_size)
    gen0.summary()
    gen1.summary()

    # build encoder models 
    input_shape = (image_size,image_size,1)
    inputs = Input(shape=input_shape,name='encoder_input')
    enc0,enc1 = build_encoder((inputs,feature1),num_labels)
    enc0.summary()
    enc1.summary()

    encoder = Model(inputs,enc1(enc0(inputs)))
    encoder.summary()

    data = (x_train,y_train),(x_test,y_test)
    train_encoder(encoder,data,model_name=model_name)
    # build adversarial model 
    optimizer = RMSprop(lr=lr*0.5,decay=decay*.5)
    # 冻结encoder 0 
    enc0.trainable = False 
    # discriminator0 weights frozen 
    dis0.trainable = False 
    gen0_inputs = [feature1,z0]
    gen0_outputs = gen0(gen0_inputs)
    adv0_outputs = dis0(gen0_outputs) + [enc0(gen0_outputs)]
    adv0 = Model(gen0_inputs,adv0_outputs,name='adv0')
    loss = ['binary_croessentropy','mse','mse']
    adv0.compile(loss=loss,loss_weights=loss_weights,optimizer=optimizer,metrics=['accuracy'])
    adv0.summary()

    # build adversariall model 
    enc1.trainable = False 
    dis1.trainable = False 
    gen1_inputs = [labels,z1]
    gen1_outputs = gen1(gen1_inputs)
    adv1_outputs = dis1(gen1_outputs)+[enc1(gen1_outputs)]
    adv1 = Model(gen1_inputs,adv1_outputs,name='adv1')
    loss_weights = [1.0,1.0,1.0]
    loss = ['binary_crossentropy','mse','categorical_crossentropy']
    adv1.compile(loss=loss,loss_weights=loss_weights,optimizer=optimizer,metrics=['accuracy'])
    adv1.summary()
    models = (enc0,enc1,gen0,gen1,dis0,dis1,adv0,adv1)
    params = (batch_size,train_steps,num_labels,z_dim,model_name)
    train(models,data,params)

def train(models,data,params):
    '''
    Train the discriminator and adversarial networks
    Discriminator is trained first with real and fake images 
    corresponding one-hot labels and latent_codes 
    Adversarial is trained next with fake images pretending 
    to be real ,corresponding one-hot labels and latent codes
    Generate sample images per save_interval 
   
    # Arguments:
        models(Models):     Encoder,Generator,Discriminator,Adversarial models 
        data(tuple):        x_train,y_train data 
        params(tuple):      Network parameters  
    '''
    enc0 ,enc1 ,gen0,gen1 ,dis0,dis1 ,adv0,adv1 = models 
    # 网络参数
    batch_size,train_steps ,num_labels ,z_dim ,model_name = params 

    (x_train,y_train),(_,_) = data 
    # the geneertor image is saved every 500 steps 
    save_interval = 500 
    
    z0 = np.random.normal(scale=0.5,size=(16,z_dim))
    z1 = np.random.normal(scale=0.5,size=(16,z_dim))
    noise_class = np.eye(num_labels)[np.arange(0,16) % num_labels]
    noise_params = [noise_class,z0,z1]
    train_size = x_train.shape[0]
    print(model_name,'Labels for generated images:',np.argmax(noise_class,axis=1))
    for i in range(train_steps):
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        real_images = x_train[rand_indexes]
        # real feature1 from encoder0 output 
        real_feature1 = enc0.predict(real_images)
        # generate random 50-dim z1 latent code 
        real_z1 = np.random.normal(scale=0.5,size=[batch_size,z_dim])

        # real labels from dataset 
        real_labels = y_train[rand_indexes]

        # generate fake feature1 
        fake_z1 = np.random.normal(sclae=0.5,size=[batch_size,z_dim])
        fake_feature1 = gen1.predict([real_labels,fake_z1])
        # real + fake data 
        feature1 = np.concatenate((real_feature1,fake_feature1))
        z1 = np.concatenate((fake_z1,fake_z1))
        # label 1st half as real and 2nd half as fake 
        y = np.ones([2*batch_size])
        y[batch_size:,:,] = 0 
        metrics =dis1.train_on_epoch(feature1,[y,z1])
        log = "%d: [dis1_loss: %f]" % (i, metrics[0])

        fake_z0 = np.random.normal(scale=0.5,size=[batch_size,z_dim])
        fake_images = gen0.predict([real_feature1,fake_z0])
        x = np.concantenate((real_images,fake_images))
        z0 = np.concantenate((fake_z0,fake_z0))
        
        metrics = dis0.train_on_batch(x, [y, z0])
        log = "%s [dis0_loss: %f]" % (log, metrics[0])
        fake_z1 = np.random.normal(scale=0.5,size=[batch_size,z_dim])
        gen1_inputs = [real_labels,fake_z1]
        y = np.ones([batch_size,1])
        metrics = adv1.train_on_epoch(gen1_inputs,[y,fake_z1,real_labels])
        fmt = "%s [adv1_loss: %f, enc1_acc: %f]"
        log = fmt%(log,metrics[0],metrics[6])

        fake_z0 = np.random.normal(scale=0.5,size=[batch_size,z_dim])
        gen0_inputs = [real_feature1,fake_z0]
        metrics = adv0.train_on_batch(gen0_inputs,
                                   [y, fake_z0, real_feature1])
        log = "%s [adv0_loss: %f]" % (log, metrics[0])
        print(log)
        if (i + 1) % save_interval == 0:
            generators = (gen0,gen1)
            # plot_images(generators)


    # gen1.save()
    # gen0.save()

