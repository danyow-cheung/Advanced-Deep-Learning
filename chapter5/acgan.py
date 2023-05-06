
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from gan import plot_images,generator,discriminator,train
from tensorflow.keras.utils import to_categorical

import numpy as np
import math
import matplotlib.pyplot as plt
import os


def build_and_train_models():
    '''加载数据集，构建lsgan鉴别器，产生器和对抗性模型'''
    (x_train,y_train),(_,_) = mnist.load_data()
    # 缩放尺寸和正则化
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train,[-1,image_size,image_size,1])
    x_train = x_train.astype('float32')/255


    # 训练类别
    num_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train)

    #
    model_name = 'acgan_mnist'
    # 网络参数
    # 潜在或z向量是100维度
    latent_size = 100
    input_shape = (image_size,image_size,1)
    label_shape = (num_labels,)
    batch_size = 64 
    lr=2e-4
    decay = 6e-8
    train_steps = 40000

    # 构建鉴别器模型
    inputs = Input(shape=input_shape,name='discriminator_input')
    discriminator = discriminator(inputs,num_labels=num_labels)
    # 使用Adam
    optimizer = RMSprop(lr=lr,decay=decay)
    loss = ['binary_crossentropy', 'categorical_crossentropy']

    # LSGAN使用MSE损失
    discriminator.compile(loss=loss,
                            optimizer=optimizer,
                            metrics = ['accuracy'],
                            )
    discriminator.summary()
    # 构建产生器模型
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape,name='z_input')
    labels = Input(shape=label_shape,name='labels')
    generator = generator(inputs,image_size,labels=labels)
    generator.summary()

    #构建对抗性模型
    optimizer = RMSprop(lr=lr*0.5,decay=decay*0.5)
    discriminator.trainable =False 
    adversarial = Model(
        [inputs,labels],
        discriminator(generator([inputs,labels])),
        name=model_name
    )

    adversarial.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()

    # 训练鉴别器和对抗性模型
    models = (generator,discriminator,adversarial)
    params = (batch_size,latent_size,train_steps,model_name)
    train(models,x_train,params)


def train(models, data, params):
    """Train the Discriminator and Adversarial Networks

    Alternately train Discriminator and Adversarial networks by batch.
    Discriminator is trained first with properly real and fake images.
    Adversarial is trained next with fake images pretending to be real
    Generate sample images per save_interval.

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        x_train (tensor): Train images
        params (list) : Networks parameters

    """
    # the GAN models
    generator, discriminator, adversarial = models
    x_train,y_train = data 

    # network parameters
    batch_size, latent_size, train_steps,num_labels, model_name = params
    # the generator image is saved every 500 steps
    save_interval = 500
    # noise vector to see how the generator output 
    # evolves during training
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    # number of elements in train dataset
    train_size = x_train.shape[0]
    noise_label = np.eye(num_labels)[np.arange(0,16)%num_labels]
    print(model_name, "Labels for generated images: ",np.argmax(noise_label, axis=1))
    for i in range(train_steps):
        # train the discriminator for 1 batch
        # 1 batch of real (label=1.0) and fake images (label=0.0)
        # randomly pick real images from dataset
        rand_indexes = np.random.randint(0,
                                         train_size, 
                                         size=batch_size)
        real_images = x_train[rand_indexes]
        real_labels = y_train[rand_indexes]
        # generate fake images from noise using generator 
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0,
                                  1.0, 
                                  size=[batch_size, latent_size])
        # generate fake images
       
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        fake_images = generator.predict([noise,fake_labels])

        # real + fake images = 1 batch of train data
        x = np.concatenate((real_images, fake_images))
        labels = np.concatenate((real_labels,fake_labels))
        # label real and fake images
        # real images label is 1.0
        y = np.ones([2 * batch_size, 1])
        # fake images label is 0.0
        y[batch_size:, :] = 0.0
        # train discriminator network, log the loss and accuracy
        metrics = discriminator.train_on_batch(x, [y, labels]) 
        fmt = "%d: [disc loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f]"
        log = fmt % (i, metrics[0], metrics[1], \
                metrics[2], metrics[3], metrics[4])
        # loss, acc = discriminator.train_on_batch(x, y)
        # log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        # train the adversarial network for 1 batch
        # 1 batch of fake images with label=1.0
        # since the discriminator weights are frozen 
        # in adversarial network only the generator is trained
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0,
                                  1.0, 
                                  size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,batch_size)]
        # label fake images as real or 1.0
        y = np.ones([batch_size, 1])
        # train the adversarial network 
        # note that unlike in discriminator training, 
        # we do not save the fake images in a variable
        # the fake images go to the discriminator input
        # of the adversarial for classification
        # log the loss and accuracy
        metrics = adversarial.train_op_batch([noise,fake_labels])
        fmt = "%s [advr loss: %f, srcloss: %f,"
        fmt += "lblloss: %f, srcacc: %f, lblacc: %f]"
        log = fmt % (log, metrics[0], metrics[1],\
                   metrics[2], metrics[3], metrics[4])
        print(log)
        if (i + 1) % save_interval == 0:
            # plot generator images on a periodic basis
            plot_images(generator,
                        noise_input=noise_input,
                        show=False,
                        step=(i + 1),
                        model_name=model_name)
   
    # save the model after training the generator
    # the trained generator can be reloaded 
    # for future MNIST digit generation
    generator.save(model_name + ".h5")




