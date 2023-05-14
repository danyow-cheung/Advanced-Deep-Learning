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
from cycleGAN import build_cyclegan,train_cycylegan 

def mnist_cross_svhn(g_models=None):
    '''
    build and train a cyclegan that can do mnist<-->svhn'''
    model_name ='cyclegan_mnist_svhn'
    batch_size = 32 
    train_steps = 100000 
    patchgan = True 
    kernel_size = 5 
    postfix = ('%dp' % kernel_size) \
               if patchgan else ('%d' % kernel_size)
    data,shapes = mnist_cross_svhn.load_data()
    source_data,_,test_source_data,test_target_data = data 
    titles = ('MNIST predicted source images.',
                 'SVHN predicted target images.',
                 'MNIST reconstructed source images.',
                 'SVHN reconstructed target images.')
    dirs = ('mnist_source-%s' \
               % postfix, 'svhn_target-%s' % postfix)
    
    # generate predicted target(svhn) and source (mnist) images 
    if g_models is not None:
        g_source,g_target = g_models
        other_utils.test_generator((g_source,g_target),
                                   (test_source_data,test_target_data),
                                   step=0,
                                   titles=titles,
                                   dirs=dirs,
                                   show=True)
        return 
    # build the cyclegan for mnist cross svhn 
    models = build_cyclegan(shapes,"mnist-%s" % postfix,
                               "svhn-%s" % postfix,
                               kernel_size=kernel_size,
                               patchgan=patchgan)
    
    # patch size is divided by 2^n since we downscaled the input in the discriminator 
    patch = int(source_data.shape[1]/2**4) if patchgan else 1 
    params = (batch_size, train_steps, patch, model_name)
    test_params = (titles, dirs)
    train_cycylegan(models,data,params,test_params,other_utils.test_generator)
    
    