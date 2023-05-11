# Cross-Domain GANS 

Cross-Domain GANS:An image in the source domain is transferred to a target domain, resulting in a new translated image.

## Principles of CycleGAN
The main disadvantage of neural networks similar to pix2pix is that the training input and output images must be aligned. Figure 7.1.1 is an example of an aligned image pair. The sample target image is generated from the source. In most occasions, aligned image pairs are not available or expensive to generate from the source images, or we have no idea on how to generate the target image from the given source image. What we have is sample data from the source and target domains. Figure 7.1.2 is an example of data from the source domain (real photo) and the target domain (Van Gogh's art style) on the same sunflower subject. The source and target images are not necessarily aligned.
与pix2pix类似的神经网络的主要缺点是，训练的输入和输出图像必须是对齐的。图7.1.1是一个对齐图像对的例子。目标图像的样本是由源图像生成的。在大多数情况下，对齐的图像对是不可用的，或者从源图像中生成的成本很高，或者我们不知道如何从给定的源图像中生成目标图像。我们所拥有的是源域和目标域的样本数据。图7.1.2是一个来自源域（真实照片）和目标域（梵高的艺术风格）的数据例子，是关于同一个向日葵的主题。源图像和目标图像不一定是对齐的。

Unlike pix2pix, CycleGAN learns image translation as long as there is a sufficient amount of, and variation between, source and target data. No alignment is needed. CycleGAN learns the source and target distributions and how to translate from source to target distribution from given sample data. No supervision is needed.
In the context of Figure 7.1.2, we just need thousands of photos of real sunflowers and thousands of photos of Van Gogh's paintings of sunflowers. After training the CycleGAN, we're able to translate a photo of sunflowers to a Van Gogh painting:
与pix2pix不同，只要源数据和目标数据之间有足够的数量和变化，CycleGAN就能学习图像翻译。不需要对齐。CycleGAN学习源和目标分布，以及如何从给定的样本数据从源到目标分布进行翻译。不需要监督。
在图7.1.2的背景下，我们只需要数千张真实向日葵的照片和数千张梵高的向日葵画作的照片。训练完CycleGAN后，我们就能把向日葵的照片翻译成梵高的画：


### The cycleGAN model 
對抗性網路的變體
The cycle consistency check implies that although we have transformed source data x to domain y, the original features of x should remain intact in y and be recoverable. The network F is just another generator that we will borrow from the backward cycle GAN, as discussed next.
循环一致性检查意味着，尽管我们已经将源数据x转化为域y，但x的原始特征应该在y中保持完整，并且可以恢复。网络F只是另一个生成器，我们将从后向循环GAN中借用，接下来将讨论。
### Implementing
