from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Dropout
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import math
from ResNet import  lr_schedule

# 加载CIFA10数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 输入图片的维度
input_shape = x_train.shape[1:]

# mormalize data
# 正则化数据
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)


# 将类向量转换为二进制类矩阵
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 定义训练参数
batch_size = 32
epochs = 200
data_augmentation=True
# 网路参数
num_classes = 10
num_dense_blocks=3
use_max_pool = False
growth_rate = 12
depth = 100
# 瓶颈层的层数
num_bottleneck_layers = (depth-4)//(2*num_dense_blocks)
num_filters_bef_dense_block = 2*growth_rate
compression_factor = 0.5

# 模型定义
#密集CNN（复合函数）由BN-ReLU-Conv2D构成
inputs = Input(shape = input_shape)
x = BatchNormalization()(inputs)
x = Activation('relu')(x)
x = Conv2D(num_filters_bef_dense_block,kernel_size=3,padding='same',kernel_initializer='he_normal')(x)
x = concatenate([inputs,x])
# 由过渡层(transition layer )桥接的致密块(dense block)堆叠
for i in range(num_dense_blocks):
    # a dense block is a stack of bottleneck layers 密集块是瓶颈层的栈
    for j in range(num_bottleneck_layers):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv2D(4*growth_rate,kernel_size=1,padding='same',kernel_initializer='he_normal')(y)

        if not data_augementation:
            y= Dropout(0.2)(y)
        x = concatenate([x,y])
    # 没有过渡层在最后一个dense模块前
    if i== num_dense_blocks -1:
        continue
        # 过渡层压缩特征图的数量并减少
        # 尺寸乘以2
    num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
    num_filters_bef_dense_block = int(num_filters_bef_dense_block*compression_factor)

    y = BatchNormalization()(x)
    y = Conv2D(num_filters_bef_dense_block,kernel_size=1,padding='same',kernel_initializer='he_normal')(y)
    if not data_augmentation:
        y = Dropout(0.2)(y)
    x= AveragePooling2D()(y)
# 在顶部添加分类器
# 平均池化后，特征图的大小为1 x 1
x= AveragePooling2D(pool_size=8)(x)
y = Flatten()(x)
outputs = Dense(num_classes,kernel_initializer='he_normal',activation='softmax')(y)

#实例化和编译模型
#原始论文使用SGD，但RMSprop对DenseNet的效果更好
model = Model(inputs=inputs,outputs=outputs)
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(1e-3),metrics=['accuracy'])
model.summary()

# 准备模型文件存储
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_densenet_model.{epoch:02d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


#为模型保存和学习速率降低准备回调

checkpoints = ModelCheckpoint(
    filepath=filepath,
    monitor='val_acc',
    verbose=1,
    save_bes_only=True
)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1),
    cooldown=0,
    patience=5,
    min_lr = 0.5e-6
)
callbacks = [checkpoints,lr_reducer,lr_scheduler]

# 跑模型
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # preprocessing  and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # 计算特征归一化所需的量
    # （如果采用ZCA增白，则标准、平均值和主要成分）
    datagen.fit(x_train)
    steps_per_epoch = math.ceil(len(x_train) / batch_size)
    model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
              verbose=1,
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)
    # 或者使用datagen.flow()进行训练
    # model.fit_generator(
    #     datagen.flow(x_train, y_train, batch_size=batch_size),
    #         steps_per_epoch=x_train.shape[0] // batch_size,
    #         validation_data=(x_test, y_test),
    #         epochs=epochs, verbose=1,
    #         callbacks=callbacks)

# score trained model
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
