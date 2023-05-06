from tensorflow.keras.layers import Dense,Input 
from tensorflow.keras.layers import Conv2D,Flatten
from tensorflow.keras.layers import Reshape,Conv2DTranspose
from tensorflow.keras.models import Model 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt 
import numpy as np
import os 

# 加载数据集
(x_train,_),(x_test,_) = mnist.load_data()

# 转换为（28,28,1)并正则化
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 网路参数
input_shape = (image_size,image_size,1)
batch_size = 32 
kernel_size = 3 
# 潜在向量
latent_dim = 16 
# 编码器/解码器 CNN的数量，过滤器
layer_filters = [32,64]
'''构建自动编码器'''
# 首先构建编码器
inputs = Input(shape=input_shape,name='encoder_input')
x = inputs 

# Conv2D(32)-Conv2D(64)栈
for filters in layer_filters:
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides =2 ,
        padding='same')(x)

#构建解码器模型所需的形状信息
#所以我们不做手工计算
#解码器的第一个输入
#Conv2DTranspose将具有此形状
#形状为（7，7，64），由
#解码器返回（28，28，1）
shape = K.int_shape(x)

# 产生潜在向量
x = Flatten()(x)
latent = Dense(latent_dim,name='latent_vector')(x)

# 实例化编码器模型
encoder = Model(inputs,latent,name='encoder')
encoder.summary()
# plot_model(encoder,
#               to_file='encoder.png',
#               show_shapes=True)
'''构建解码器'''
latent_inputs = Input(shape=(latent_dim,),name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
#从矢量到合适形状的变换变换
x = Reshape((shape[1],shape[2],shape[3]))(x)

# Conv2DTranspose(64)--Conv2DTranspose(32) 栈
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=2,
        padding='same')(x)

#重构输入
outputs = Conv2DTranspose(
    filters =1,
    kernel_size=kernel_size,
    activation='sigmoid',
    padding='same',
    name='decoder_output')(x)

# 实例化解码器
decoder = Model(latent_inputs,outputs,name='decoder')
decoder.summary()
# plot_model(decoder, to_file='decoder.png', show_shapes=True)

# autoencoder = encoder + decoder
# 实例化自编码器
autoencoder = Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()
# plot_model(autoencoder,
#            to_file='autoencoder.png',
#            show_shapes=True)

autoencoder.compile(loss='mse',optimizer='adam')
# 训练模型
autoencoder.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=1,batch_size=batch_size)
# 从测试数据中预测
x_decoded = autoencoder.predict(x_test)
# 展示环节
# display the 1st 8 test input and decoded images
imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('input_and_decoded.png')
plt.show()


