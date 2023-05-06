from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import os

def rgb2gray(rgb):
    '''把RGB模式转换为灰度图'''
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])

# 加载数据
(x_train,_),(x_test,_) = cifar10.load_data()
# 输入图片维度
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]
# 创建saved_images 文件夹

imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(),imgs_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
# 展示1st 100 输入图片
imgs = x_test[:100]
imgs = imgs.reshape((10,10,img_rows,img_cols,channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

# 将彩色的训练和测试转换为灰色
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

# display grayscale version of test images
# 展示测试图片的灰度图
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()

# 正则化输出训练和测试颜色图片
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255 

# 正则化输入训练和测试灰度图
x_train_gray = x_train_gray.astype('float32')/255
x_test_gray = x_test_gray.astype('float32')/255 

# 将图像重塑为行x列x通道，用于CNN输出/验证
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,channels)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,channels)

# 将图像重塑为 rowxcolxchannel 用于CNN输入
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0],img_rows,img_cols,1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0],img_rows,img_cols,1)

# 网络参数
input_shape = (img_rows,img_cols,1)
batch_size = 32
kernel_size = 3
latent_dim = 256
#每层CNN层和滤波器的编码器/解码器数量
layer_filters = [64,128,256]

# 构建自动编码器模型
"''首先构建编码器''"
inputs = Input(shape=input_shape,name='encoder_input')
x = inputs
# Conv2D(64)-Conv2D(128)-Conv2D(256) 栈
for filters in layer_filters:
    x = Conv2D(filters = filters,kernel_size=kernel_size,strides=2,activation='relu',padding='same')(x)
#构建解码器模型所需的形状信息，因此我们不需要手动操作
# 计算
#解码器的第一个Conv2Transpose的输入将具有
# 形状
#形状是（4，
# 32, 3)
shape = K.int_shape(x)
# 产生潜在向量
x = Flatten()(x)
latent = Dense(latent_dim,name='latent_vector')(x)
# 初始化编码器模型
encoder = Model(inputs,latent,name='encoder')
encoder.summary()

'''构建解码器模型'''
latent_inputs = Input(shape=(latent_dim,),name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1],shape[2],shape[3]))(x)

# Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64) 栈
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=2,activation='relu',padding='same')(x)

outputs = Conv2DTranspose(filters=channels,kernel_size=kernel_size,activation='sigmoid',name='decoder_output')(x)
# 初始化解码器
decoder = Model(latent_inputs,outputs,name='decoder')
decoder.summary()

'''autoencoder = encodr + decoder'''
'''初始化自动编码器'''
autoencoder = Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()

# 准备模型的保存文件夹
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# 降低学习率通过sqrt（0.1) 如果loss在5个epoch里没有提升
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1),
    cooldown=0,
    patience=5,
    verbose=1,
    min_lr = 0.5e-6
)

# 保存模型的权重
checkpoints = ModelCheckpoint(filepath=filepath,monitor='val_loss',verbose=1,save_best_only=True)

# 模型编译
autoencoder.compile(loss='mse',optimizer='adam')

# 每个epoch都回调一次
callbacks = [lr_reducer,checkpoints]
# 训练模型
autoencoder.fit(
    x_train_gray,
    x_train,
    validation_data=(x_test_gray,x_test),
    epochs=30,
    batch_size=batch_size,
    callbacks=callbacks)

# 预测模型输出
x_decoded = autoencoder.predict(x_test_gray)

# 展示图片
imgs = x_decoded[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png' % imgs_dir)
plt.show()