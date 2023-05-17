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
    