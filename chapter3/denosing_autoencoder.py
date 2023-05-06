from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# 加载数据集
(x_train,_),(x_test,_) = mnist.load_data()

# 转换为（28,28,1)并正则化
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 通过添加具有正常距离的噪声来生成损坏的MNIST图像
# 以0.5为中心，std=0.5
noise = np.random.normal(loc=0.5,scale=0.5,size=x_train.shape)
x_train_noisy = x_train + noise

noise = np.random.normal(loc=0.5,scale=0.5,size=x_test.shape)
x_test_noisy = x_test+noise

# 添加噪声可能超过标准化像素值>1.0或？<0.0
#剪辑像素值>1.0到1.0且<0.0到0.0
x_train_noisy= np.clip(x_train_noisy,0.,1.)
x_test = np.clip(x_test_noisy,0.,1.)
# 网络参数
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
# 实例化自编码器
autoencoder = Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()


autoencoder.compile(loss='mse',optimizer='adam')
autoencoder.fit(
    x_train_noisy,x_train,
    validation_data =(x_test_noisy,x_test),
    epochs=10,
    batch_size=batch_size
)

#从损坏的测试图像预测自动编码器输出
x_decoded = autoencoder.predict(x_test_noisy)


#3组9 MNIST数字的图像
#第一行-原始图像
#第2行-图像被噪声损坏
#第3行-去噪图像
rows, cols = 3, 9
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
             'Corrupted Input: middle rows, '
             'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()