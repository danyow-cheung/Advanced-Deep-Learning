# Disentangled Representation GANS 分離表示 GANS
换句话说，GANs还可以学习解纠缠的潜在代码或表示，我们可以使用这些代码或表示来改变生成器输出的属性。解纠缠的代码或表示是一种张量，可以改变输出数据的特定特征或属性，同时不影响其他属性。

In summary, the goal of this chapter is to present:
- The concepts of disentangled representations
- The principles of both InfoGAN and StackedGAN
- Implementation of both InfoGAN and StackedGAN using tf.keras
- -解纠缠表示的概念
- InfoGAN和StackedGAN的原理
- 使用tf.keras实现InfoGAN和StackedGAN

## InfoGAN
> InfoGAN.py

## StackedGAN 
> stackedGAN.py
本著與 InfoGAN 相同的精神，StackedGAN 提出了一種方法來解開調節生成器輸出的潛在表示。 然而，StackedGAN 使用不同的方法來解決這個問題
**Instead of learning how to condition the noise to produce the desired output, StackedGAN breaks down a GAN into a stack of GANs. Each GAN is trained independently in the usual discriminator-adversarial manner with its own latent code.**
StackedGAN 不是學習如何調節噪聲以產生所需的輸出，而是將 GAN 分解為一堆 GAN。 每個 GAN 都以通常的鑑別器對抗方式使用自己的潛在代碼進行獨立訓練。
