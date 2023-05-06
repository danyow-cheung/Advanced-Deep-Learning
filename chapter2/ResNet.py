'''
在CIRFAR10数据集上训练残差网络RedNet
'''
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import math


def lr_schedule(epoch):
    '''学习率规划
    @param
        epoch :epoch的数量
    @retun
        lr:学习率
    '''
    lr = 1e-3 
    if epoch >180:
        lr *= 0.5e-3 
    elif epoch > 160:
        lr *= 1e-3 
    elif epoch > 120:
        lr *= 1e-2 
    elif epoch > 80:
        lr *= 1e-1 
    print("leanring rate:",lr)
    return lr 
    
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):
  '''2D Convolution-Batch Normalization-Activation stack builder
  @param
    inputs:来自输入图像或前一层的输入张量
    num_filters:Conv2D过滤器数量
    kernel_size:COnv2D内核尺寸
    strides:Conv2D步长
    activation:激活函数
    batch_normalization:是否包括批处理规范化
    conv_first:检测是conv-bn-activation还是bn-activatin-conv
  @return
    x:张量作为下一层的输入

  '''
  conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer = l2(1e-4))
  x = inputs
  if conv_first:
    x = conv(x)
    if batch_normalization:
      x = BatchNormalization()(x)
    if activation is not None:
      x = Activation(activation)(x)
  else:
    if batch_normalization:
      x= BatchNormalization()(x)
    if activation is not None:
      x = Activation(activation)(x)
    x = conv(x) 
  return x 

def resnet_v1(input_shape,depth,num_classess=10):
    """
    ResNet Version 1 Model builder [a]ResNet版本1模型生成器
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU 2 x（3 x 3）Conv2D BN ReLU堆叠

        Last ReLU is after the shortcut connection. 最后一个ReLu在shortcut连接后，
        At the beginning of each stage, the feature map size is halved在每个阶段开始时，特征图大小减半
        (downsampled) by a convolutional layer with strides=2, while （下采样）由步幅为2的卷积层进行，而
        the number of filters is doubled. Within each stage, 滤波器的数量加倍。在每个阶段内，
        the layers have the same number filters and the same number of filters. 图层具有相同的过滤器数量
        -------------------
        Features maps sizes:特征图尺寸
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        ------------------
        @param
            input_shape (tensor): shape of input image tensor输入图像张量的形状
            depth (int): number of core convolutional layers核心卷积层的数量
            num_classes (int): number of classes (CIFAR10 has 10)类的数量（CIFAR10有10个）
        @return 
            model (Model): Keras model instance Keras模型实例
    """

    if (depth-2)%6!=0:
        raise ValueError("深度应该是6n+2(比如20,32 )")
    # 模型定义
    num_filters = 16 
    num_res_blocks = int((depth-2)/6)

    inputs = Input(shape=input_shape) 
    x = resnet_layer(inputs=inputs)
    # 变量残差集合stack
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1 
            # 第一层但不是第一个堆
            if stack > 0 and res_block ==0:
                strides = 2 #下采样
            y = resnet_layer(inputs = x,num_filters=num_filters,strides=strides)

            y = resnet_layer(inputs=y,num_filters=num_filters,activation=None)
            # 第一层但不是第一个堆
            if stack > 0 and res_block==0:
                # 线性残差shortcut
                # 连接去匹配改变的dims
                x = resnet_layer(inputs=x,num_filters = num_filters,kernel_size=1,strides=strides,activation=None,batch_normalization=False)

            x = add([x,y])
            x = Activation('relu')(x)
        num_filters *=2 

    # 添加分类在顶部
    # v1 在最后的shortcut 连接relu，不使用BN，
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classess,activation='softmax',kernel_initializer='he_normal')(y)
    #初始化模型
    model = Model(inputs=inputs,outputs=outputs)
    return model 


def resnet_v2(input_shape,depth,num_classess=10):
    '''ResNet version2 模型构建器
    （1 x 1）-（3 x 3）-（1 x 2）BN-ReLU-Conv2D或
    也称为瓶颈层。
    每层的第一个快捷连接(shortcut connection)是1 x 1 Conv2D。
    第二个及以后的快捷连接是身份(identity)。
    在每个阶段开始时，
    特征图大小减半（下采样）
    通过strides＝2的卷积层，
    而过滤器映射的数量为
    加倍。在每个阶段中，层具有
    相同数量的过滤器和相同的过滤器映射大小。
    -------------------
    特征图尺寸
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    -------------------
    @param
        input_shape:输入图片的张量
        depth:conv层数
        num_classess:类别数量
    @return
        model:keras模型
    '''
    if (depth -2 )%9!=0:
        raise ValueError("深度应该是9n+2(比如110)")
    num_filters_in = 16 
    num_res_blocks = int((depth-2)/9)
    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs,num_filters=num_filters_in,conv_first=True)

    # 实例化剩余单元堆栈
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation ='relu'
            batch_normalization=True 
            strides = 1 
            if stage ==0:
                num_filters_out = num_filters_in*4
                if res_block==0:
                    activation = None 
                    batch_normalization =False 
            else:
                num_filters_out = num_filters_in*2 
                if res_block:
                    strides =2 
            # 瓶颈剩余单元
            y = resnet_layer(
                inputs = x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False )
            
            y = resnet_layer(inputs=y,num_filters=num_filters_in,conv_first=False)

            y = resnet_layer(inputs=y,num_filters=num_filters_in,conv_first=False)

            if res_block==0:
                # 线性投影剩余捷径连接
                #以匹配更改的尺寸
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False)
            x = add([x,y])
        num_filters_in = num_filters_out
    # 在顶部添加分类器。
    #v2在池化之前具有BN ReLU
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classess,activation='softmax',kernel_initializer='he_normal')(y)

    # 初始化模型
    model = Model(inputs=inputs,outputs=outputs)
    return model 

n = 3 
# 模型版本
version = 1 

# 计算深度从模型的参数n中
if version ==1:
  depth = n*6 +2 
elif version==2:
  depth = n*9+2


# 模型名字，深度和版本
model_type = 'ResNet%dv%d'%(depth,version)

# 训练参数
batch_size = 32 
epochs = 200  
data_augmentation=True 
num_classes =10 

# 像素平均值减法提高精度
subtract_pixel_mean = True 


# 加载数据集
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# 输入图片的维度
input_shape = x_train.shape[1:]
# 正则化
x_train = x_train.astype("float32")/255
x_test = x_test.astype('float32')/255 

# 如果启用了像素平均值减法
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train,axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
# #将类向量转换为二进制类矩阵。
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

if version == 2:
  model = resnet_v2(input_shape=input_shape,depth=depth)
elif version==1:
  model = resnet_v1(input_shape=input_shape,depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['acc'])
model.summary()
plot_model(model, to_file=f"{model_type}.png" , show_shapes=True)
print(model_type)

#保存模型
save_dir = os.path.join(os.getcwd(),'save_models')
model_name = 'cirfar10_%s_model.{epoch:03d}.h5'%model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir,model_name)

# 为模型保存和学习率调整准备回调。
checkpoints = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
callbacks = [checkpoints,lr_reducer,lr_scheduler]

# 跑模型使用与否数据增强
if not data_augmentation:
    print("不使用数据增强")
    model.fit(x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test,y_test),
    shuffle=True,
    callbacks=callbacks)
else:
    print("使用数据增强")
    # 这个api要了解一下
    datagen = ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False ,
        featurewise_std_normalization=False ,

        samplewise_std_normalization= False ,
        zca_whitening=False ,
        
        rotation_range = 0,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip=True,
        vertical_flip = False 
    )
    datagen.fit(x_train)

    steps_per_epoch = math.ceil(len(x_train)/batch_size)
    # fit the model on the batches generated by datagen.flow().
    model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
              verbose=1,
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)


# 评估模型
scores = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])