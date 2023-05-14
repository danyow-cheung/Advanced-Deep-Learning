# Variational Autoencoderes (VAES)變分自動編碼器

The generator of VAEs is able to produce meaningful outputs while navigating its continuous latent space. The possible attributes of the decoder outputs are explored through the latent vector.
VAE 的生成器能夠在導航其連續潛在空間時產生有意義的輸出。 通過潛在向量探索解碼器輸出的可能屬性。

在結構方面，VAE 與自動編碼器有相似之處。 它還由編碼器（也稱為識別或推理模型）和解碼器（也稱為生成模型）組成。 VAE 和自動編碼器都試圖在學習潛在向量的同時重建輸入數據。


In the same line of discussions on GANs that we discussed in the previous chapters, the VAE's decoder can also be conditioned. 
For example, in the MNIST dataset, we're able to specify the digit to produce given a one-hot vector. This class of conditional VAE is called **CVAE**[2]. VAE latent vectors can also be disentangled by including a regularizing hyperparameter on the loss function. This is called **ß-VAE** [5].
For example, within MNIST, we're able to isolate the latent vector that determines the thickness or tilt angle of each digit. 
The goal of this chapter is to present:

todo:
1. The principles of VAE 
2. An understanding of the reparameterization trick that facilitates the use 
   of stoschastic gradients on VAE optimization

3. The principles of conditional VAE(CVAE) optimization
4. An understanding of how to implement VAE using tf.keras


## Principles of VAE 

### Variational inference
### Core equation
### optimiztion
### Reparameterization trick 
### Decoder testing 
### VAE in keras 
### using CNN for AE 

## Conditional VAE(CVAE)

## -VAE-VAE without disentangled latent repressentations 

## COnclusion

## References 
