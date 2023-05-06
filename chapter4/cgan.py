
'''有了CGAN，就像有一个代理人，
我们可以要求他画出类似于人类书写数字的数字。
CGAN相对于DCGAN的关键优势在于，
我们可以指定我们希望代理绘制的数字。
'''
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization,concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from dcgan import plot_images,build_and_train_models

import numpy as np
import os 
import matplotlib.pyplot as plt 
import math 

'''
对发生器和鉴别器输入施加条件。条件是数字的一个热矢量版本。
这与要生成（生成器）或分类为真实或伪（鉴别器）的图像相关。
'''
'''对抗性模型是通过串联生成器和鉴别器来实现的'''

def build_discriminator(inputs,labels,image_size):
    '''构建鉴别器
    LeakyReLU-Conv2D的堆栈，以区分真假。
    @param参数列表
        inputs:(layer)鉴别器的输入层
        labels:(layer)一个热向量的输入层，用于调节输入
        image_size:one side s的目标尺寸（假设为方形图像）
    @return返回值
        discriminator:(Model)产生鉴别器模型
    '''

    kernel_size = 5 
    layer_filters = [32,64,128,256]
    
    x = inputs 
    '''cgan改变'''
    y = Dense(image_size*image_size)(labels)
    y = Reshape((image_size,image_size,1))(y)
    x = concatenate([x,y])

    for filters in layer_filters:
        # 前三层卷积使用strides=2
        # 最后一层卷积使用strides=1
        if filters == layer_filters[-1]:
            strides = 1 
        else:
            strides = 2 
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides = strides,
            padding='same'
        )(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    '''cgan改变'''
    discriminator = Model([inputs,labels],x,name='discriminator')
    return discriminator



def build_generator(inputs,labels,image_size):
    '''构建生成器模型
    BN-ReLU-Conv2DTranspose堆栈以生成假图像
    输出激活是Sigmoid的，而不是[1]中的tanh。
    sigmoid易于收敛
    @param参数列表
        inputs:(Layer)生成器的输入
        image_size:(tensor)一侧的目标尺寸（假设为方形图像）
        labels:(layer)一个热向量的输入层，用于调节输入
    @return返回值
        generator:(Model)产生模型
    '''

    image_size = image_size//4
    # 网络参数
    kernel_size = 5 
    layer_filters = [128,64,32,1]

    x = concatenate([inputs,labels],axis=1)

    x = Dense(image_size*image_size*layer_filters[0])(inputs)
    x= Reshape((image_size,image_size,layer_filters[0]))(x)

    for filters in layer_filters:
        # 前两层卷积使用strides =2 
        # 最后两层使用strides = 1 
        if filters>layer_filters[-2]:
            strides = 2 
        else:
            strides = 1 
        x= BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(
            filters = filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same'
        )(x)

    x = Activation('sigmoid')(x)

    generator = Model([inputs,labels],x,name='generator')
    return generator



def train(models,data,params):
    '''训练鉴别器和对抗性网络
    
    分批交替训练鉴别和对抗网络。
    鉴别器首先使用正确的真实图像和伪图像进行训练。
    对抗者接下来会用假图像假装真实进行训练
    每个save_interval生成示例图像。
    
    @param参数列表
        models(list):生成器鉴别器和对抗性模型
        x_train(tensor):训练图像
        params(list):网路参数
    '''
    # GAN模型组成
    generator,discriminator ,adversarial = models
    # 图像和标签
    x_train,y_train = data 

    # 网络参数
    batch_size,latent_size,train_steps,num_labels,model_name = params 
    # 每500个阶段保存生成器的图片
    save_interval = 500 
    # 噪声矢量，查看在训练中的generator输出
    noise_input = np.random.uniform(-1.0,1.0,size=[16,latent_size])
    #一个热标签噪音将被调节
    noise_class = np.eye(num_labels)[np.arange(0,16) % num_labels]
    
    # 训练数据集中的元素数目
    train_size = x_train.shape[0]
    
    print(model_name,
             "Labels for generated images: ",
             np.argmax(noise_class, axis=1))

    for i in range(train_steps):
        # 训练鉴别器一个epoch
        #1 epoch of 真实（标签=1.0）和假图像（标签=0.0）
        # 从数据集中随机选取真实图像
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        real_images = x_train[rand_indexes]
        '''cgan change'''
        # 对应真实图像的一个热标签
        real_labels = y_train[rand_indexes]

        # 产生假的图片从噪声的生产器
        # 使用均匀分布生成噪声
        noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

        '''cgan change'''
        # 指派random的独热编码
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,batch_size)]

        #生成以假标签为条件的假图像
        fake_images = generator.predict([noise,fake_labels])

        # 产生假的图片
        # fake_images = generator.predict(noise)
        
        
        # real+fake images = 1 batch of train data 
        x = np.concatenate((real_images,fake_images))
        '''cgan changes'''
        # real + fake one-hot labels = 1 batch of train one- hot labels
        labels = np.concatenate((real_labels,fake_labels))


        # 给图片打标签
        # label real and fake images 
        # real label is 1
        y = np.ones([2*batch_size,1])
        # fake images label is 0
        y[batch_size:,:,] = 0.0 
        # 训练鉴别器网络，加载损失和精确度
        '''cgan change'''
        # loss,acc = discriminator.train_on_batch(x,y)
        loss,acc = discriminator.train_on_batch([x,labels],y)

        log = '%d: [discriminator loss :%f, acc: %f]'%(i,loss,acc)
        
        # 训练对抗性模型for 1 epoch
        #1批标签为1.0的假图像
        #因为鉴别器权重
        #被冻结在对抗网络中
        #只有发电机经过培训
        #使用均匀分布生成噪声
        noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

        '''cgan changes'''
        # 随机分配一个热标签
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,batch_size)]

        # label fake images as real or 1
        y = np.ones([batch_size,1])
        #训练对抗网络
        #注意，与鉴别器训练不同，
        #我们不会将假图像保存在变量中
        #假图像进入对抗性的
        #用于分类
        #记录损失和准确性
        # loss,acc = adversarial.train_on_batch(noise,y)
        loss, acc = adversarial.train_on_batch([noise, fake_labels], y)

        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i+1)% save_interval==0:
            # 定期绘制生成器图像
            plot_images(generator,noise_input=noise_input,show=False,step=(i+1),model_name=model_name)
    # 保存模型在生成器完成之后
    generator.save(model_name+'.h5')

if __name__ =='__main__':
    build_and_train_models()