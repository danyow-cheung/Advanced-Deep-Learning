"""Code implementation of Policy Gradient Methods as solution
to MountainCarCountinuous-v0 problem

Methods implemented:
    1) REINFORCE
    2) REINFORCE with Baseline
    3) Actor-Critic
    4) A2C
"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import plot_model
import tensorflow_probability as tfp 

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import argparse
import gym
from gym import wrappers, logger
import csv
import time
import os
import datetime
import math


def softplusk(x):
    '''
    Some implementations use a modified softplus 
    to ensure that the stddev is never zero
    Argument
        x(tensor):  acitvation input 
    '''
    return K.softplus(x) + 1e-10


class PolicyAgent:
    def __init__(self,env) -> None:
        '''
        Implement the models and trianing of Policy gradient methods
        Argument
            env(object)     Openai gym enviroment
        '''
        self.env = env 
        # entropy loss weights 
        self.beta = 0.0 
        # value loss for all policy graditns except A2C 
        self.loss = self.value_loss 

        # s,a,r,s' are stored in memory 
        self.memory = []
        # for compuation of input size 
        self.state = env.reset()
        self.state_dim = env.observation_space.shape[0]
        self.state = np.reshape(self.state,[1,self.state_dim])
        self.build_autoencoder()
    
    def build_autoencoder(self):
        '''autoencoder to convert states into features'''
        # fist build the encoder model 
        inputs = Input(shape=(self.state_dim,),name='state')
        feature_size = 32 
        x = Dense(256,activation='relu')(inputs)
        x = Dense(128,activation='relu')(x)
        feature = Dense(feature_size,name='feature_vector')(x)

        # instantiate encoder model 
        self.encoder = Model(inputs,feature,name='encoder')
        self.encoder.summary()
        # plot_model(self.encoder,to_file='en')
        # build the decoder model 
        feature_inputs = Input(shape=(feature_size,),name='decoder_input')
        x = Dense(128, activation='relu')(feature_inputs)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self.state_dim, activation='linear')(x)

        # instiantiate decoder model 
        self.decoder = Model(feature_inputs,outputs,name='decoder')
        self.decoder.summary()

        # autoencoder = encoder + decoder 
        self.autoencoder = Model(inputs,self.decoder(self.encoder(inputs)),name='autoencoder')
        self.autoencoder.summary()

        # MEAN square error(MSE) loss function Adam optimzer 
        self.autoencoder.compile(loss='mse',optmizer='adam')

    def train_autoencoder(self,x_train,x_test):
        '''
        Training the autoencoder using randomly samples states from the env 
        Argument
            x_train(tesnor)     autoencoder trian dataset  
            x_test(tensor)      autoencoder test dataset 
        '''
        batch_size = 32 
        self.autoencoder.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=10,batch_size=batch_size)
    
    def action(self,args):
        '''
        Given mean and stddev,sample an action clip and resturn 
        We assume Gaussian distributioin of probability of selecting an action given a state
        Argument
            args(list)  mean ,stddev,list
        '''
        mean,stddev = args 
        dist = tfp.distributions.Normal(loc=mean, scale=stddev)
        action = dist.sample(1)
        action = K.clip(action,self.env.action_space.low[0],self.env.action_space.high[0])
        return action
    





class REINFORCEAgent(PolicyAgent):
    def __init__(self,env):
        '''
        Implements the models and training of REINFORCE policy gradient method 
        Arguments:
            env(object):    OpenAI gym enviroment
        '''
        super().__init__(env)
    
    def train_by_episode(self):
        '''
        Train by episode
        Prepare the dataset before the step by step training 
        '''
        rewards = []
        gamma = 0.99 
        for item in self.memory:
            [_,_,_,reward,_]=item
            rewards.append(reward)
        
        # compute return per step 
        for i in range(len(rewards)):
            reward = rewards[i:]
            horizon = len(reward)
            discount = [math.pow(gamma,t) for t in range(horizon)]
            return_ = np.dot(reward,discount)
            self.memory[i][3] = return_ 
        # train every step 
        for item in self.memory:
            self.train(item,gamma=gamma)
        
    def train(self,item,gamma=1.0):
        '''Main routine for trianing 
        Arguments:
            item(list):     one experience unit
            gamma(float):   discount factor[0,1]
        '''
        [step,state,next_state,reward,done] = item 
        # must save state for entropy computation
        self.state = state 

        discount_factor = gamma*step 
        delta = reward
        
        # apply the discount factor as shown in Algorithms
        discounted_delta = delta *discount_factor
        discounted_delta = np.reshape(discounted_delta,[-1,1])
        verbose = 1 if done else 0 

        # train the logp model 
        self.logp_model.fit(np.array(state),discounted_delta,batch_size=1,epochs=1,verbose=verbose)


class REINFORCEBaselineAgent(REINFORCEAgent):
    def __init__(self, env):
        '''
        Implements the method and training of REINFORCE w/ baseline policy gradient method
        Arguments:
            env(object):    OpenAI gym enviroment
        '''
        super().__init__(env)
    

    def train(self,item,gamma=1.0):
        '''
        Main routine for training 
        Arguments:
            item(list):     one experience unit 
            gamma(float):   discount factor[0,1]

        '''
        [step,state,next_state,reward,done] = item 
        # must save state for entropy computation 
        self.state = state 
        
        discount_factor = gamma * step
        # reinforce -baseline:delta = return - value 
        delta = reward - self.value(state)[0]

        # apply the discount factor as shown in Algorithm
        discount_delta = delta * discount_factor
        discount_delta = np.reshape(discount_delta,[-1,1])
        verbose = 1 if done else 0 

        # trian the logop model 
        self.logp_model.fit(np.array(state),discount_delta,batch_size=1,epochs=1,
                            verbose=verbose)
        # train the value network 
        self.value_model.fit(np.array(state),discount_delta,batch_size=1,epochs=1,verbose=verbose)

        

class ActorCriticAgent(PolicyAgent):
    def __init__(self,env):
        '''
        Implements the models and training of Actor Critic policy gradient method 
        Arguments;
            env(object):    OpenAi gym enviroment
        '''
        super().__init__(env)

    def trian(self,item,gamma=1.0):
        '''
        Main routine for training 
        Arguments:
            item(list)          one experience unit
            gamma(float)        discount factor[0,1]
        '''
        [step,state,next_state,reward,done] = item 
        # must save state for entropy computation
        self.state = state 

        discount_factor = gamma**step 

        delta = reward - self.value(state)[0]

        #since this function is called by Actor-Critic 
        #directly,evaluate the value function here 
        if not done:
            next_value = self.value(next_state)[0]
            delta += gamma * next_value
        # apply the discount factor as shown in Algortihms
        discount_delta = delta * discount_factor
        discount_delta = np.reshape(discount_delta,[-1,1])
        verbose = 1 if done else 0 

        #  trian the logp model 
        self.logp_model.fit(
            np.array(state),
            discount_delta,
            batch_size=1,
            epochs=1,
            verbose=verbose
        )



class A2CAgent(PolicyAgent):
    def __init___(self,env):
        '''
        Implements the models and trianing of A2C training of A2C policy gradient method 
        Arguments:
            env(Object):       OpenAI gym enviroment
        '''
        super().__init__(env)
        # beta of entropy used in a2c 
        self.beta = 0.9 
        # loss functon of A2C value_model is mse 
        self.loss = 'mse'
    
    def trian_by_episode(self,last_value=0):
        '''
        Trian by episode
        Prepare the dataset before the step training 
        Argument
            last_value(float)   previous prediction of value state 
        '''
        #implements a2c trianing from the last state to the first state 
        # discount factor 
        gamma = 0.95 
        r = last_value 
        # the memory is visited in reverse as shown 
        for item in self.memory[::-1]:
            [step,state,next_state,reward,done] = item 
            r = reward + gamma*r 
            item = [step,state,next_state,r,done]
            # trian per step 
            self.train(item)
    
    def train(self,item,gamma=1.0):
        '''Main routine for training 
        Argument
            item(list)      one experience unit 
            gamma(float)    discount factor[0,1]
        '''
        [step,state,next_state,reward,done] = item 
        # must save state for entropy computations
        self.state = state 

        discount_factor = gamma**step 
        # a2c:delat = 
        delta = reward - self.value(state)[0]

        verbose = 1 if done else 0 
        self.logp_model.fit(np.array(False),discount_factor,batch_size=1,epochs=1,verbose=verbose)
        # in A2C ,the target value is the return (reward replaced by return in the train_by_episodes)
        discount_delta = reward 
        discount_delta = np.reshape(discount_delta,[-1,1])

        self.value_model.fit(np.array(state),discount_delta,batch_size=1,epochs=1,verbose=verbose)
