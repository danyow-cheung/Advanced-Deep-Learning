'''Y-Network两个输入，使用concatenate进行连接'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Dropout,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,concatenate
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
# 加载数据集
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# 计算类别
num_labels = len(np.unique(y_train))
# 转换为独热编码
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 输入图片维度
image_size = x_train.shape[1]
# 更改图片尺寸和归一化
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 网络参数
input_shape = (image_size,image_size,1)
batch_size = 32
kernel_size= 3
n_filters = 32
dropout = 0.4
# 左边的分支
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layers(32-64-128)
for i in range(3):
  x = Conv2D(filters = filters,kernel_size =kernel_size,padding='same',activation='relu')(x)
  x = Dropout(dropout)(x)
  x = MaxPooling2D()(x)
  filters *=2
# 右边分支
right_inputs = Input(shape = input_shape)
y = right_inputs
filters = n_filters
for i in range(3):
  y = Conv2D(filters = filters,kernel_size =kernel_size,padding='same',activation='relu')(y)
  y = Dropout(dropout)(y)
  y = MaxPooling2D()(y)
  filters *=2

# 将两边进行连接
Y = concatenate([x,y])
# 在连接到Dense之前的特征图转换为矩阵
Y = Flatten()(y)

Y = Dropout(dropout)(Y)
outputs = Dense(num_labels,activation='softmax')(Y)
# 建立模型
model = Model([left_inputs,right_inputs],outputs)
# 使用画图来验证模型
plot_model(model,to_file='cnn-y-network.png',show_shapes =True)
# 使用summary验证
model.summary()

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the model with input images and labels
model.fit([x_train, x_train],
          y_train,
          validation_data=([x_test, x_test], y_test),
          epochs=20,
          batch_size=batch_size)
# model accuracy on test dataset
score = model.evaluate([x_test, x_test],
                       y_test,
                       batch_size=batch_size,
                       verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
