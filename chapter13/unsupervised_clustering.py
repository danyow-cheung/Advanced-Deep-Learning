
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

import numpy as np
import os
import argparse
import vgg

from data_generator import DataGenerator
from utils import unsupervised_labels, center_crop
from utils import AccuracyCallback, lr_schedule


class IIC:
    def __init__(self,args,backbone):
        '''
        Contains the encoder model,the loss function,loading of datasets ,train and evaluation
        routines to implement IIC unsupervised clustering via mutual information maximization
        Arguments:
            args:Command line arguments to indicate choice of batch size,number of heads,folder todave 
                weights file,weights file name etc
            s
            backbone(Model)     IIC encoder backbone(eg VGG)
            
        '''
        self.args = args 
        self.backbone = backbone
        self._model = None 
        self.train_gen = DataGenerator(args,simaese=True)
        self.n_labels = self.train_gen.n_labels 
        self.build_model()
        self.load_eval_dataset()
        self.accuracy = 0 

    
    def build_model(self):
        '''
        Build the n_heads of the IIC model
        '''
        inputs = Input(shape=self.train_gen.input_shape,name='x')
        x = self.backbone(inputs)
        x = Flatten()(x)
        # number of output heads 
        outputs = []
        for i in range(self.args.heads):
            name = 'z_head%d'%i 
            outputs.append(Dense(self.n_labels,activation='softmax',name=name)(x))

        self._model = Model(inputs,outputs,name='encoder')
        optimizer = Adam(lr=1e-3)
        self._model.compiile(optimizer=optimizer,loss=self.mi_loss)
    

    def mi_loss(self,y_true,y_pred):
        '''
        Mutal information loss computed from the joint 
        distribution to matrix and the marginals 
        Arguments:
            y_true(tensor)      No used since this is unsupervised learning
            y_pred(tensor)      stack of softmax predictions for the siamese latent vectors
            
        '''
        size = self.args.batch_size 
        n_labels = y_pred.shape[-1]
        # lower half is Z 
        Z = y_pred[0:size,:]
        Z = K.expand_dims(Z,axis=2)
        #upper half is Zbar 
        Zbar = y_pred[size:y_pred.shape[0],:]
        Zbar = K.expand_dims(Zbar,axis=1)

        # compute joint distribution
        P = K.batch_dot(Z,Zbar)
        P = K.sum(P,axis=0)
        # enforce symmetric joint distribution 
        P = (P + K.transpose(P)) / 2.0

        # normalization of total probability to 1.0 
        P = P/K.sum(P)
        
        # marginal distributions 
        Pi = K.expand_dims(K.sum(P,axis=1),axis=1)
        Pj = K.expand_dims(K.sum(P,axis=0),axis=0)
        Pi = K.repeat_elements(Pi, rep=n_labels, axis=1)
        Pj = K.repeat_elements(Pj, rep=n_labels, axis=0)
        P = K.clip(P, K.epsilon(), np.finfo(float).max)
        Pi = K.clip(Pi, K.epsilon(), np.finfo(float).max)
        Pj = K.clip(Pj, K.epsilon(), np.finfo(float).max)

        # negative MI loss 
        neg_mi = K.sum(P*(K.log(Pi)+K.log(Pj) - K.log(P)))

        # each head contribute 1/n_heads to the total loss 
        return neg_mi/self.args.heads 
        
    def train(self):
        '''Train function uses the data generator,
        accuracy computation and learning rate 
        scheduler callbacks
        '''
        accuracy = AccuracyCallback(self)
        lr_schedule = LearningRateScheduler(lr_schedule,verbose=1)
        callbacks =[accuracy,lr_schedule]
        self._model.fit_generator(
            generator=self.train_gen,
            use_multiprocessing=True,
            epochs=self.args.epochs,
            callbacks=callbacks,
            workers=4,
            shuffle=True
        )
    
    def eval(self):
        '''
        Evlaute the accuracy of the current model weights
        '''
        y_pred = self._model.predict(self.x_test)
        print('')
        # accuracy per head 
        for head in range(self.args.heads):
            if self.args.heads ==1:
                y_pred = y_pred
            else:
                y_head = y_pred[head]
            y_head = np.argmax(y_head,axis=1)
            accuracy = unsupervised_labels(
                list(self.y_test),
                list(y_head),
                self.n_labels,
                self.n_labels,
            )
            info = "Head %d accuracy: %0.2f%%"
            if self.accuracy>0:
                info += ", Old best accuracy: %0.2f%%"
                data = (head, accuracy, self.accuracy)
            else:
                data = (head,accuracy)
            print(info % data)

            if accuracy>self.accuracy and self.args.save_weights is not None:
                self.accuracy = accuracy 
                folder = self.args.save_dir 
                os.makedirs(folder,exist_ok=True)
                path = os.path.join(folder, self.args.save_weights)
                print('Saving weights',path)
                self._model.save_weights(path)

            