from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from gan import plot_images,generator,discriminator,train
import numpy as np

def build_and_train_models():
    '''加载数据集，构建lsgan鉴别器，产生器和对抗性模型'''
    (x_train,_),(_,_) = mnist.load_data()
    # 缩放尺寸和正则化
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train,[-1,image_size,image_size,1])
    x_train = x_train.astype('float32')/255
    #
    model_name = 'lsgan_mnist'
    # 网络参数
    # 潜在或z向量是100维度
    latent_size = 100
    input_shape = (image_size,image_size,1)
    batch_size = 64 
    lr=2e-4
    decay = 6e-8
    train_steps = 40000

    # 构建鉴别器模型
    inputs = Input(shape=input_shape,name='discriminator_input')
    discriminator = discriminator(inputs,activation=None)
    # 使用Adam
    optimizer = RMSprop(lr=lr,decay=decay)
    # LSGAN使用MSE损失
    discriminator.compile(loss='mse',
                            optimizer=optimizer,
                            metrics = ['accuracy'],
                            )
    discriminator.summary()
    # 构建产生器模型
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape,name='z_input')
    generator = generator(inputs,image_size)
    generator.summary()

    #构建对抗性模型
    optimizer = RMSprop(lr=lr*0.5,decay=decay*0.5)
    discriminator.trainable =False 
    adversarial = Model(
        inputs,
        discriminator(generator(inputs)),
        name=model_name
    )
    adversarial.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()

    # 训练鉴别器和对抗性模型
    models = (generator,discriminator,adversarial)
    params = (batch_size,latent_size,train_steps,model_name)
    train(models,x_train,params)



