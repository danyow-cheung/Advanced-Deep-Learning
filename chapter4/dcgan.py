from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import os 
import matplotlib.pyplot as plt 
import math 


def build_and_train_models():
    # 加载数据集
    (x_train,_),(_,_) = mnist.load_data()

    # 转换为（28,28,1)并正则化
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train,[-1,image_size,image_size,1])

    x_train = x_train.astype('float32') / 255

    model_name = 'dcgan_mnist'
    # 网路参数
    # 潜在或z矩阵is 100-dim
    latent_size = 100 
    batch_size = 64 
    train_steps =  40000
    lr = 2e-4 
    decay = 6e-8

    input_shape = (image_size,image_size,1)
    '''构建鉴别器模型'''
    inputs = Input(shape=input_shape,name='discriminator_input')
    discriminator = build_discriminator(inputs)
    
    # original paper use Adam
    ##但鉴别器容易与RMSprop收敛
    optimizer = RMSprop(lr=lr,decay=decay)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics = ['accuracy'])
    discriminator.summary()

    '''构建产生器模型'''
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape,name='z_input')
    generator = build_generator(inputs,image_size)
    generator.summary()
    
    '''构建对抗性模型'''
    optimizer = RMSprop(lr=lr*0.5,decay=decay*0.5)
    #在对抗性训练中冻结鉴别器的权重
    discriminator.trainable = False 

    # 对抗性 = 生成器+鉴别器
    adversarial = Model(
        inputs,
        discriminator(generator(inputs)),
        name=model_name
    )
    adversarial.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    adversarial.summary()

    # 训练鉴别器和对抗性模型
    models = (generator,discriminator,adversarial)
    params = (batch_size,latent_size,train_steps,model_name)
    '''由于自定义训练，通常的fit（）函数将不会被使用。
    相反，调用train_on_batch（）为给定的数据批运行一次梯度更新。'''
    train(models,x_train,params)


def build_generator(inputs,image_size):
    '''构建生成器模型
    BN-ReLU-Conv2DTranspose堆栈以生成假图像
    输出激活是Sigmoid的，而不是[1]中的tanh。
    sigmoid易于收敛
    @param参数列表
        inputs:(Layer)生成器的输入
        image_size:(tensor)一侧的目标尺寸（假设为方形图像）
    @return返回值
        generator:(Model)产生模型
    '''
    image_size = image_size//4
    # 网络参数
    kernel_size = 5 
    layer_filters = [128,64,32,1]

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
    generator = Model(inputs,x,name='generator')
    return generator



'''对抗性模型是通过串联生成器和鉴别器来实现的'''

def build_discriminator(inputs):
    '''构建鉴别器
    LeakyReLU-Conv2D的堆栈，以区分真假。
    @param参数列表
        inputs:(layers)鉴别器的输入层
    @return返回值
        discriminator:(Model)产生鉴别器模型
    '''
    kernel_size = 5 
    layer_filters = [32,64,128,256]
    x = inputs 
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
    discriminator = Model(inputs,x,name='discriminator')
    return discriminator

'''完成后，生成器将生成一批新的假图像，并标记为真实（1.0）。
该批次将用于训练对抗网络。两个网络交替训练约40000步。
每隔一段时间，基于某个噪声向量生成的MNIST数字就会保存在文件系统中。
在最后一个训练步骤中，网络已经聚合。生成器模型也保存在一个文件中，
因此我们可以很容易地重用经过训练的模型，以便将来生成MNIST数字。
然而，仅保存生成器模型，因为这是该DCGAN在生成新MNIST数字时的有用部分。
例如，我们可以通过执行以下操作生成新的随机MNIST数字：'''
def train(models,x_train,params):
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
    # 网络参数
    batch_size,latent_size,train_steps,model_name = params 
    # 每500个阶段保存生成器的图片
    save_interval = 500 
    # 噪声矢量，查看在训练中的generator输出
    noise_input = np.random.uniform(-1.0,1.0,size=[16,latent_size])

    # 训练数据集中的元素数目
    train_size = x_train.shape[0]
    
    for i in range(train_steps):
        # 训练鉴别器一个epoch
        #1 epoch of 真实（标签=1.0）和假图像（标签=0.0）
        # 从数据集中随机选取真实图像
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        real_images = x_train[rand_indexes]

        # 产生假的图片从噪声的生产器
        # 使用均匀分布生成噪声
        noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

        # 产生假的图片
        fake_images = generator.predict(noise)
        # real+fake images = 1 batch of train data 
        x = np.concatenate((real_images,fake_images))

        # 给图片打标签
        # label real and fake images 
        # real label is 1
        y = np.ones([2*batch_size,1])
        # fake images label is 0
        y[batch_size:,:,] = 0.0 
        # 训练鉴别器网络，加载损失和精确度
        loss,acc = discriminator.train_on_batch(x,y)
        log = '%d: [discriminator loss :%f, acc: %f]'%(i,loss,acc)
        
        # 训练对抗性模型for 1 epoch
        #1批标签为1.0的假图像
        #因为鉴别器权重
        #被冻结在对抗网络中
        #只有发电机经过培训
        #使用均匀分布生成噪声
        noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

        # label fake images as real or 1
        y = np.ones([batch_size,1])
        #训练对抗网络
        #注意，与鉴别器训练不同，
        #我们不会将假图像保存在变量中
        #假图像进入对抗性的
        #用于分类
        #记录损失和准确性
        loss,acc = adversarial.train_on_batch(noise,y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i+1)% save_interval==0:
            # 定期绘制生成器图像
            plot_images(generator,noise_input=noise_input,show=False,step=(i+1),model_name=model_name)
    # 保存模型在生成器完成之后
    generator.save(model_name+'.h5')


def plot_images(generator,noise_input,show=False,step=0,model_name='gan'):
    '''生成假图像并绘制它们
    出于可视化目的，生成假图像
    然后在正方形网格中绘制它们
    
    @param参数列表
    generator(模型):generator模型假图像生成
    noise_input(ndarray):z向量数组
    show(bool):是否显示绘图
    step(int):附加到保存图像的文件名
    model_name(string):模型名称
    '''
    os.makedirs(model_name,exist_ok =True)
    filename = os.path.join(model_name,'%05d.png'%step)
    images = generator.predict(noise_input)
    plt.figure(figsize = (2.2,2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


def test_generator(generator):
    noise_input = np.random.uniform(-1.0,1.0,size=[16,100])
    plot_images(generator,noise_input=noise_input,show=True,model_name='test_outputs')


if __name__ =="__main__":
    # parser = argparse.ArgumentParser()
    # help_ = "Load generator h5 model with trained weights"
    # parser.add_argument("-g", "--generator", help=help_)
    # args = parser.parse_args()
    # if args.generator:
    #     generator = load_model(args.generator)
    #     test_generator(generator)
    # else:
    #     build_and_train_models()
    build_and_train_models()

