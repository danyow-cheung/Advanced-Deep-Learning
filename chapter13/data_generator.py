
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np
from skimage.transform import resize, rotate



class DataGenerator(Sequence):
    def __init__(self,
                 args,
                 shuffle=True,
                 siamese=False,
                 mine=False,
                 crop_size=4):
        """Multi-threaded data generator. Each thread reads
            a batch of images and performs image transformation
            such that the image class is unaffected

        Arguments:
            args (argparse): User-defined options such as
                batch_size, etc
            shuffle (Bool): Whether to shuffle the dataset
                before sampling or not
            siamese (Bool): Whether to generate a pair of 
                image (X and Xbar) or not
            mine (Bool): Use MINE algorithm instead of IIC
            crop_size (int): The number of pixels to crop
                from all sides of the image
        """
        self.args = args
        self.shuffle = shuffle
        self.siamese = siamese
        self.mine = mine
        self.crop_size = crop_size
        self._dataset()
        self.on_epoch_end()
    

    def __data_generation(self,start_index, end_index):
        '''
        Data generation algorithm. The method generates
               a batch of pair of images (original image X and
               transformed imaged Xbar). The batch of Siamese
               images is used to trained MI-based algorithms:
               1) IIC and 2) MINE (Section 7)
        Arguments:
            start_index(int)    Given an array of images,
                   this is the start index to retrieve a batch
            end_index(int)      Given an array of images,
                   this is the end index to retrieve a batch

        '''
        d = self.crop_size//2 
        crop_sizes = [self.crop_size*2+i for i in range(0,5,2)]

        image_size = self.data.shape[1] - self.crop_size
        
        x = self.data[self.indexes[start_index,end_index]]
        y1 = self.label[self.indexes[start_index:end_index]]
        
        target_shape = (x.shape[0],*self.input_shape)
        x1 = np.zeros(target_shape)
        if self.siamese:
            y2 = y1 
            x2 = np.zeros(target_shape)
        
        for i in range(x1.shape[0]):
            image = x[i]
            x1[i] = image[d:image_size +d,d:image_size+d]
            if self.siamese:
                rotate = np.random.randint(0,2)
                # 50%-50% chance of crop or rotate 
                if rotate==1:
                    shape = target_shape[1:]
                    x2[i] = self.random_rotate(image,target_shape=shape)
                else:
                    x2[i] = self.random_crop(image,target_shape[1:],crop_sizes)
        
        # for IIC we are mostly interested in paired images
        # X and Xbae = G(x)
        if self.siamese:
            # if MINE algorithm is chosen,use this to generate the training data 
            if self.mine:
                y  = np.concatenate([y1,y2],axis=0)
                m1 = np.copy(x1)
                m2 = np.copy(x2)
                np.random.shuffle(m2)

                x1 = np.concatenate((x1,m1),axis=0)
                x2 = np.concatenate((x2,m2),axis=0)
                x = (x1,x2)
                return x,y 
            x_train = np.concatenate([x1,x2],axis=0)
            y_train = np.concatenate([y1,y2],axis=0)
            y = []
            for i in range(self.args.heads):
                y.append(y_train)
            return x_train,y 
    
        return x1,y1 
    
