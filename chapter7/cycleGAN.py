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
    x = inputs 
    if instance_norm:
        x = InstanceNormalization(axis=3)(x)
    if activation =='relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x,paired_inputs])
    return x 

def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):
    '''
    The generator is a U-Network made of a 4-layer encoder 
    and a 4-layer decoder.Layer n-i is connected to layer i 
    Arguments:
        input_shape(tuple):     input shape
        output_shape(tuple):    output shape 
        kernel_size(int):       kernel size of encoder & decoder layers 
        name(string):           name assigned to generator model 
    Returns:
        generator(Model):
    '''
    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs,32,kernel_size=kernel_size,activation='leakly_relu',strides=1)
    e2 = encoder_layer(e1,64,activation='leaky_relu',kernel_size=kernel_size)
    e3 = encoder_layer(e2,128,activation='leaky_relu',kernel_size=kernel_size)
    e4 = encoder_layer(e3,256,activation='leaky_relu',kernel_size=kernel_size)
    d1 = decoder_layer(e4,e3,128,kernel_size=kernel_size)

    d2 = decoder_layer(d1,e2,64,kernel_size=kernel_size)

    d3 = decoder_layer(d2,e1,32,kernel_size=kernel_size)

    outputs = Conv2DTranspose(channels,kernel_size=kernel_size,strides=1,activation='relu',padding='same')(d3)
    generator = Model(inputs,outputs,name=name)
    return generator



def build_discriminator(input_shape,
                        kernel_size=3,
                        patchgan=True,
                        name=None):
    '''
    The discriminator is a 4-layer encoder that outputs either 
    a 1-dim or a n x n-dim patch of probability that input is real 
    Arguments:
        inputs_shape(tuple):        input shape
        kernel_size(int):           kernel size of decoder layers 
        patchgan(bool):             whether the output is a patch or just a 1-dim 
        name(string):               name assigned to discriminator model
    Returns:
        discriminator(Model):
    '''
    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs,
                    32,
                    kernel_size=kernel_size,
                    activation='leaky_relu',
                    instance_norm=False)
    
    x = encoder_layer(x,
                    64,
                    kernel_size=kernel_size,
                    activation='leaky_relu',
                    instance_norm=False)
    
    x = encoder_layer(x,128,kernel_size=kernel_size,activation='leaky_relu',instance_norm=False)

    x = encoder_layer(x,256,kernel_size=kernel_size,strides=1,activation='leaky_relu',instance_norm=False)
    

    # if patchgan = True use nxn-dim output of probability
    # else use 1-dim output of probability
    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1,kernel_size,strides=2,padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)
    discriminator = Model(inputs,outputs,name=name)
    return discriminator

def build_cyclegan(shapes,source_name='source',
                   target_name='target',kernel_size=3,
                   patchgan=False,identity=False):
    '''
    Build the CycleGAN
    1) Build target and source discriminator 
    2) Build target and source generators 
    3) Build the adversarial network
    Arguments:
        shape(tuple):           source and target shapes
        source_name(string):    string to be appended on dis/gen models 
        target_name(string):    string to be append on dis/gen models 
        kernel_size(int):       kernel size for the encoder/decoder or dis/gen models 
        patchgan(bool):         whether to use patchgan on discriminator
        identity(bool):         whether to use identity loss
    Returns:
        (list):                 2 generator ,2 discriminator ,and 1 adversarial models 
    '''
    source_shape ,target_shape= shapes 
    lr = 2e-4 
    decay = 6e-8 
    gt_name = "gen_" + target_name
    gs_name = "gen_" + source_name
    dt_name = "dis_" + target_name
    ds_name = "dis_" + source_name
    # build target and source generators 
    g_target = build_generator(
        source_shape,target_shape,kernel_size=kernel_size,
        name=gt_name
    )
    g_source = build_generator(
        target_shape,source_shape,kernel_size=kernel_size,name=gs_name
    )
    g_target.summary()
    g_source.summary()
    # build target and source discriminators 
    d_target = build_discriminator(target_shape,patchgan=patchgan,kernel_size=kernel_size,name=dt_name)
    d_source = build_discriminator(source_shape,patchgan=patchgan,kernel_size=kernel_size,name=ds_name)
    d_target.summary()
    d_source.summary()
    optimizer = RMSprop(lr=lr,decay=decay)
    d_target.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
    d_source.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
    d_target.trainable = False 
    d_source.trainable = False 

    # build the computational graph for the adversarial model 
    source_input = Input(shape=source_shape)
    fake_target = g_target(source_input)
    preal_target = d_target(fake_target)
    reco_source = g_source(fake_target)

    # backward cycle network and source discriminator 
    target_input = Input(shape=target_shape)
    fake_source = g_source(target_input)
    preal_source = d_source(fake_source)
    reco_target = g_target(fake_source)

    # if we ues identity loss add 2 extra loss terms  and outpus 
    if identity:
        iden_source = g_source(source_input)
        iden_target = g_target(target_input)
        loss = ['mse','mse','mae','mae','mae','mae']
        loss_weights = [1.0,1.0,10.,10.,0.5,0.5]
        inputs = [source_input,target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target,
                   iden_source,
                   iden_target
                   ]
    else:
        loss = ['mse','mse','mae','mae']
        loss_weights = [1.0,1.0,10.,10.]
        inputs = [source_input,target_input]
        outputs =  [preal_source,
                   preal_target,
                   reco_source,
                   reco_target,
                   ]
    # build adversarial model 
    adv = Model(inputs,outputs,name='adversarial')
    optimizer = RMSprop(lr=lr*0.5,decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    
    adv.summary()
    return g_source,g_target,d_source,d_target,adv 


def train_cycylegan(models,
                    data,
                    params,
                    test_params,
                    test_generator):
    '''
    Train the cycle gan 
    1) train the target discriminator 
    2) train the source discriminator 
    3) train the forward and backward cycles of adversarial networks
    Arguments:
        models (Models): Source/Target Discriminator/Generator,
            Adversarial Model
        data (tuple): source and target training data
        params (tuple): network parameters
        test_params (tuple): test parameters
        test_generator (function): used for generating
            predicted target and source images
    '''
    # the models 
    g_source,g_target ,d_source ,d_target ,adv = models 
    # network params 
    batch_size,train_steps,patch,model_name = params
    # train dataset 
    source_data,target_data,test_source_data,test_target_data = data 
    titles,dirs = test_params
    # the generator image is saved every 2000 steps 
    save_interval = 2000 
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]
    # whether to use pathchgan or not 
    if patch>1:
        d_patch = (patch,patch,1)
        valid = np.ones((batch_size,)+d_patch)
        fake = np.ones((batch_size,)+d_patch)
    else:
        valid = np.ones([batch_size,1])
        fake = np.onse([batch_size,1])
    valid_fake = np.concanate((valid,fake))
    start_time = datetime.datetime.now()
    for step in range(train_steps):
        # sample a batch of real target data 
        rand_indexes = np.random.randint(0,target_size,size=batch_size)
        real_target = target_data[rand_indexes]
        # sample a batch of real source data 
        rand_indexes = np.random.randint(0,source_size,size=batch_size)
        real_source = source_data[rand_indexes]
        
        # generate a batch of fake target data fr real source data 
        fake_target = g_target.predict(real_source)
        # compile real and fake into one batch 
        x = np.concanate((real_target,fake_target))
        # train the target discriminator using fake/real data 
        metrics = d_target.train_on_epoch(x,valid_fake)
        log = "%d: [d_target loss: %f]" % (step, metrics[0])

        # generate a batch of fake source data fr real target data 
        fake_source = g_source.predict(real_target)
        x = np.concanate((real_source,fake_source))
        # train the source discriminator using fake/real data 
        metrics = d_source.train_on_batch(x,valid_fake)
        log = '%s[d_source loss:%f]'%(log,metrics[0])

        # train the adversarial network using forward and backward cycles 
        # the generated fake source and target data attempts to trick the 
        # discriminators 
        x = [real_source,real_target]
        y = [valid,valid,real_source,real_target]
        metrics = adv.trian_on_batch(x,y)
        elapsed_time = datetime.datetime.now()-start_time
        fmt = "%s [adv loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)
        if (step+1)%save_interval ==0:
            test_generator((g_source,g_target),(test_source_data,test_target_data),
                           step=step+1,
                           titles=titles,
                           dirs=dirs,
                           show=False)
            

def graycifar10_cross_colorcifar10(g_models=None):
    '''
    build and train a cyclegan that can do grayscale<-->color cifar10 images
    '''
    model_name = 'cyclgan_cifar10'
    batch_size = 32 
    train_steps = 10000
    patchgan = True 
    kernel_size = 3 
    postfix = ('%dp' % kernel_size) \
               if patchgan else ('%d' % kernel_size)
    

    data,shapes = cifar10_utils.load_data()
    source_data,_ ,test_source_data,test_target_data = data 
    titles = ('CIFAR10 predicted source images.',
                 'CIFAR10 predicted target images.',
                 'CIFAR10 reconstructed source images.',
                 'CIFAR10 reconstructed target images.')
    dirs = ('cifar10_source-%s' % postfix, 
               'cifar10_target-%s' % postfix)
    # generate predicted target(color) and source (gray) images 
    if g_models is not None:
        g_source,g_target = g_models
        other_utils.test_generator((g_source,g_target),
                                   (test_source_data,test_target_data),
                                   step=0,
                                   titles=titles,
                                   dirs=dirs,
                                   show=True)

        return 
    # build the cyclegan for cifar10 colorization 
    models = build_cyclegan(shapes,
                             "gray-%s" % postfix,
                               "color-%s" % postfix,
                               kernel_size=kernel_size,
                               patchgan=patchgan)
    
    # patch size is divided by 2^n since we downscaled the input 
    # in the discriminator by 2^n (ie we use strides =2 n times)
    patch = int(source_data.shape[1]/2**4) if patchgan else 1 
    params = (batch_size,train_steps,patch,model_name)
    test_params = (titles,dirs)
    # train 
    train_cycylegan(
        models,
        data,
        test_params,
        other_utils.test_generatpr
    )


