'''
用Keras在MNIST上训练WGAN
使用Wassertein损耗训练GAN。与DCGAN相似的是
输出的线性激活和n_critical训练的使用
对抗训练。鉴别器权重被裁剪为
Lipschitz约束的要求。
'''
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from gan import plot_images,generator,discriminator


import numpy as np
def wasserstein_loss(y_label,y_pred):
    return -K.mean(y_label*y_pred)

def train(models,x_train,params):
    '''
    训练Discriminator网络和对抗网络分批交替训练Discriminator网络和对抗网络。
    首先用正确标记的真实和虚假图像训练Discriminator。鉴别器权重按要求被裁剪
    利普希茨约束。生成器接下来(通过对抗性)与
    假装真实的假图像。
    每个save_interval生成示例图像
    @param参数列表
        model(list):生成器，鉴别器，敌对的模型
        x_train(张量):训练图像
        params (list):网络参数
    '''
    # GAN模型
    generator,discriminator,adversarial = models
    # 网络参数
    (batch_size,latent_size,n_critic,clip_value,trian_steps,model_name) = params
    # 生产器图像没500张保存一次
    save_interval = 500
    # 噪声对生成器的
    noise_input = np.random.uniform(-1.0,1.0,size=[16,latent_size])
    #number of elements in train dataset
    train_size = x_train.shape[0]
    # 标签对真是数据
    real_labels = np.ones((batch_size,1))
    for i in range(trian_steps):
        loss = 0
        acc = 0
        for _ in range(n_critic):
            # 训练一个epoch的鉴别器
            # 1 batch of real and fake images
            # 随机从数据集中选择真实图片
            rand_indexes = np.random.randint(0,train_size,size=batch_size)
            real_images = x_train[rand_indexes]

            # 产生假的图片从噪音中
            # 产生噪声使用使用均匀分布
            noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])

            fake_images = generator.predictt(noise)
            # 训练鉴别器网络
            # real data label=1
            # fake data label=-1
            #而不是1批次的真实和虚假的图像，先用1批真实数据，再用1批假图片。
            #这个调整防止渐变（梯度）
            #因对立而消失
            #标记的真实和虚假的数据标签(即+1和-1)和
            # 由于剪切，重量的幅度很小。
            real_loss,real_acc = discriminator.train_on_batch(real_images,real_labels)
            fake_loss,fake_acc = discriminator.train_on_batch(fake_images,-real_labels)

            # #累积平均损失和精度
            loss += 0.5 * (real_loss+fake_loss)
            acc += 0.5 * (real_acc+fake_acc)
            # 翻转鉴别器的权重满足Lipschitz约束
            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(weights,-clip_value,clip_value) for weight in weights]
                layer.set_weights(weights)
        # 每次n_批评家训练迭代的平均损失和准确性
        loss /= n_critic
        acc /= n_critic
        log = '%d:[discriminator loss:%f,acc:%f]'%(i,loss,acc)
        # 训练对抗网络1批次
        # 第一批标签=1.0的假图片
        # 因为鉴别器的权重被冻结在
        #对抗网络只有生成器受过训练
        #使用均匀分布产生噪声
        noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        #训练对抗网络
        #注意，与辨别训练不同，
        #我们不保存伪图像在一个变量
        #假图片交给鉴别者
        #输入对抗性分类
        #假图片被贴上真实的标签
        #记录损失和精度
        loss,acc= adversarial.train_on_batch(noise,real_labels)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)

        if (i+1)%save_interval==0:
            ## plot生成器图像的周期
            
            plot_images(
                generator,
                noise_input=noise_input,
                show=False,
                step=(i+1),
                model_name=model_name
            )
    
    #在训练生成器后保存模型
    #训练过的生成器可以重新加载
    #用于未来的MNIST数字生成
    generator.save(model_name+'.h5')

def build_and_train_models():
    '''加载数据集，构建WGAN鉴别器，产生器和对抗性模型
    调用WGAN训练
    '''
    # 加载数据集
    (x_train,_),(_,_) = mnist.load_data()
    # 缩放尺寸到（28,28,1）和正则化
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train,[-1,image_size,image_size,1])
    x_train = x_train.astype('float32')/255
    model_name = 'wgan_minst'
    # 网络参数
    # 潜在或者z向量是100维度
    latent_size = 100
    # 超参数
    n_critic = 5
    clip_value = 0.01
    batch_size = 64
    lr = 5e-5
    train_steps = 40000
    input_shape = (image_size,image_size,1)

    # 构建鉴别器模型
    inputs = Input(shape=input_shape,name='discriminator_input')
    # WGAN使用线性激活
    discriminator = discriminator(inputs,activation='linear')

    optimizer = RMSprop(lr=lr)
    # WGAN鉴别器使用wassertein_loss
    discriminator.compile(
        loss=wasserstein_loss,
        optimizer=optimizer,
        metrics = ['accuracy'],
    )
    discriminator.summary()

    # 构建产生器模型
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape,name='z_input')
    generator = generator(inputs,image_size)
    generator.summary()

    # 构建对抗性模型
    # 释放鉴别器的权重在训练对抗性模型的时候
    discriminator.trainable =False
    adversarial = Model(
        inputs,
        discriminator(generator(inputs)),
        name=model_name
    )
    adversarial.compile(loss=wasserstein_loss,
                        optimizer=optimizer,
                        metrics = ['accuracy']
    )
    adversarial.summary()
    # 训练模型
    models = (generator,discriminator,adversarial)
    params = (batch_size,
              latent_size,
              n_critic,
              clip_value,
              train_steps,
              model_name)
    train(models,x_train,params)

if __name__ =='__main__':
    build_and_train_models()