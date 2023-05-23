from tensorflow.keras.layers import Input, Dense, Add, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
import os
import argparse
import vgg

import matplotlib.pyplot as plt
from scipy.stats.contingency import margins
from data_generator import DataGenerator
from utils import unsupervised_labels, center_crop, AccuracyCallback, lr_schedule


def sample(joint=True,mean=[0,0],conv=[[1,0.5],[0.5,1]],n_data=1000000):
    '''
    Helper function to obtain samples fr a bivariate Gaussian distribution
    Arguments;
        joint(bool)     if joint distribution is desired
        mean(list)      the mean values of the 2d gaussian
        cov(list)       the convariance matrix of the 2d gaussian
        n_data(int)     number of samples fr 2d gaussian
    '''
    xy = np.random.multivariate_normal(mean=mean,conv=conv,size=n_data)
    # samples fr joint distribution 
    if joint:
        return xy 
    y = np.random.multivariate_normal(mean=mean,cov=conv,size=n_data)

    # samples fr marginal distriburion 
    x = xy[:,0].reshape(-1,1)
    y = y[:,1].reshape(-1,1)
    xy =np.concatenate([x,y],axis=1)
    return xy 
def compute_mi(cov_xy=0.5,n_bins=100):
    '''
    Ananlytic computation of MI using binned 2D gaussian
    Arguments:
        cov_xy(list)    off-diagonal elements of convariance matrix 
        n_bins(int)     number of bins to quantize the continuous 2d gaussian
    '''
    cov = [[1,cov_xy],[cov_xy,1]]
    data =sample(cov=cov)
    # get joint distribution samples 
    # perform histogtam binning 
    joint,edge = np.histogram(data,bins=n_bins)
    joint /= joint.sum()
    eps = np.finfo(float).eps 
    joint[joint<eps]=eps 
    
    x,y = margins(joint)
    xy = x*y 
    xy[xy<eps]=eps 
    # MI is P(x,y)*log(p(x,y)/p(x)*p(y))
    mi = joint*np.log(joint/xy)
    mi = mi.sum()
    return mi 


class SimpleMINE:
    def __init__(self,args,input_dim=1,hidden_units=16,output_dim=1) -> None:
        '''
        Learn to compute MI using MINE
        Arguments:
            args        User-defined arguments such as off-diagonal elements of convariance matrix,batch size,epochs,etc
            input_dim(int)      input size dimension 
            hiddent_units(int)  number of hidden units of the MINE MLP network 
            output_dim(int)     output size dimension 
        '''
        self.args = args 
        self._model = None
        self.build_model(input_dim,hidden_units,output_dim)
    
    def build_model(self,input_dim,hidden_units,output_dim):
        '''
        Build a simple MINE model 
        Arguments;
            see class arguments
        '''
        inputs1 = Input(shape=(input_dim),name='x')
        inputs2 = Input(shape=(input_dim),name='y')
        x1 = Dense(hidden_units)(inputs1)
        x2 = Dense(hidden_units)(inputs2)
        x = Add()([x1,x2])
        x = Activation('relu',name='ReLU')(x)
        outputs  = Dense(output_dim,name='MI')(x)
        inputs = [inputs1,inputs2]
        self._model = Model(inputs,outputs,name='MINE')
        self._model.summary()
    
    def mi_loss(self,y_true,y_pred):
        '''
        MINE loss function
        Arguments
            y_true(tensor)  Not used since this is unsupervised learning 
            y_pred(tensor)  stack of predictions for joint T(x,y) and marginal T(x,y)'''
        size = self.args.batch_size 
        # lower half is pred for joint dist 
        pred_xy = y_pred[0:size,:]
        
        # upper half is pred for marginal dist 
        pred_x_y = y_pred[size:y_pred.shape[0],:]
        # implentation of MINE loss 
        loss = K.mean(pred_xy)-K.log(K.mean(K.exp(pred_x_y)))
        return -loss

    def train(self):
        '''Train the MNIE to estimate MI between X and Y of a 2D Gaussian'''
        optimizer = Adam(lr=0.01)
        self._model.compile(optimizer=optimizer,loss=self.mi_loss)
        plot_loss = []
        cov = [[1,self.args.cov_xy],[self.args.cov_xy,1]]
        loss = 0. 
        for epoch in range(self.args.epochs):
            # joint dist samples 
            xy = sample(n_data=self.args.batch_size,cov=cov)
            x1 = xy[:,0].reshape(-1,1)
            y1 = xy[:,1].reshape(-1,1)
            # marginal dist samples 
            xy = sample(joint=False,n_data=self.args.batch_size,cov=cov)
            x2 = xy[:,0].reshape(-1,1)
            y2 = xy[:,1].reshape(-1,1)

            # train on batch of joint & marginal samples 
            x = np.concatenate((x1,x2))
            y = np.concatenate((y1,y2))

            loss_item = self._model.train_on_epoch([x,y],np.zeros(x.shape))
            loss += loss_item
            plot_loss.append(-loss_item)
            if (epoch + 1) % 100 == 0:
                fmt = "Epoch %d MINE MI: %0.6f"
                print(fmt % ((epoch+1), -loss/100))
                loss = 0.


class MINE:
    def __init__(self,args,backone) -> None:
        '''
        Contains the encoder,SimpleMINE and linear classifier models 
        .the loss function,loadding of datasets,train and evaluation routines
        to implement MINE unsupervised clustering via mutual information maximization
        Arguments:
            args:
            backbone(Model)     MINE Encoder backbone (eg VGG)
        '''
        self.args = args 
        self.latent_dim = args.latent_dim
        self.backbone = backone
        self._model = None 
        self._encoder = None 
        self.train_gen = DataGenerator(args,siamese=True,mine=True)
        self.n_labels = self.train_gen.n_labels 
        self.build_model()
        self.accuracy = 0 
    
    def build_model(self):
        '''
        Build the MINE model unsupervised classifier 
        '''
        inputs = Input(shape=self.train_gen.input_shape,name='x')
        x = self.backbone(inputs)
        x = Flatten()(x)

        y = Dense(self.latent_dim,activation='linear',name='encoded_x')(x)
        # encoder is based on backbone 
        # feature extractor 
        self._encoder = Model(inputs,y,name='encoder')
        # the simpleMINE is bivariate Gaussian is used 
        self._mine = SimpleMINE(
            self.args,
            input_dim=self.latent_dim,
            hidden_units=1024,
            output_dim=1
        )
        inputs1 = Input(shape=self.train_gen.input_shape,name='x')
        inputs2 = Input(shape=self.train_gen.input_shape,name='y')
        x1 = self._encoder(inputs1)
        x2 = self._encoder(inputs2)
        outputs = self._mine.model([x1,x2])
        self._model = Model([inputs1,inputs2],outputs,name='encoder')
        optimizer = Adam(lr=1e-3)
        self._model.compile(optimizer=optimizer,loss=self.mi_loss)
        self._model.summary()
        self.load_eval_dataset()
        self._classifier = LinearClassifier(
            latent_dim=self.latent_dim
        )

    def train(self):
        '''
        Train MINE to estimate MI between X and Y 
        '''
        accuracy = AccuracyCallback(self)
        lr_scheduler = LearningRateScheduler(lr_schedule,verbose=1)

        callbacks = [accuracy,lr_scheduler]
        self._model.fit_generator(generator=self.train_gen,use_multiprocessing=True,
                                  epochs=self.args.epochs,
                                  callbacks=callbacks,
                                  workers=4,
                                  shuffle=True)
        
    def eval(self):
        '''
        Evaluate the accuracy of the current model weights
        '''
        y_pred = self._encoder.predict(self.x_test)
        self._classifier.train(y_pred,self.y_test)
        accuracy = self._classifier.eval(y_pred,self.y_test)
        
        info = "Accuracy: %0.2f%%"
        if self.accuracy>0:
            info += ", Old best accuracy: %0.2f%%"
            data = (accuracy,self.accuracy)
        else:
            data =(accuracy)
        print(info%data)
        if accuracy > self.accuracy \
        and self.args.save_weights is not None:
            folder = self.args.save_dir
            os.makedirs(folder, exist_ok=True)
            args = (self.latent_dim, self.args.save_weights)
            filename = "%d-dim-%s" % args
            path = os.path.join(folder, filename)
            print("Saving weights... ", path)
            self._model.save_weights(path)
        if accuracy>self.accuracy:
            self.accuracy = accuracy


class LinearClassifier:
    def __init__(self,latent_dim=10,n_classes=10):
        '''
        A simple MLP-based linear classifier 
        A linear clssifier is an MLP network without non-linear activation like ReLU 
        This can be used as a substitute to linear assignment algorithm
        Arguments:
            latent_dim(int)     Latent vector dimensionality
            n_classess(int)     Number of classess the latent dim will be converted to
        '''
        self.buid_model(latent_dim,n_classes)
    
    def build_model(self,latent_dim,n_classes):
        ''' Linear classifier model builder.
        
        '''
        inputs = Input(shape=(latent_dim,),name='cluster')
        x = Dense(256)(inputs)
        outputs = Dense(n_classes,activation='softmax',name='class')(x)
        name ='classifier'
        self._model = Model(inputs,outputs,name=name)
        self._model.compile(loss='categorical_crossentropy',optimizers='adam',metrics=['accuracy'])
        self._model.summary()
        
