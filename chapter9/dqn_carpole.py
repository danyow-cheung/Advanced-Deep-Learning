from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger

class DQAgent:
    def __init__(self,state_space,action_space,episodes=500) -> None:
        '''
        DQN Agent on CartPole-v0 env
        Arguments:
            state_space(tensor)     state space
            action_space(tensor)    action space
            episodes(int)           number of episodes to train
        '''
        self.action_space = action_space

        # experience buffer 
        self.memory = []

        # discount rate 
        self.gamma = 0.9 

        # initiailly 90% exploration 10% exploitation 
        self.epsilon = 1.0 
        # iteratively applying decay til 
        # 10% exploration /90% exploitation
        self.epsilon_min = 0.1 
        self.epsilon_decay = self.epsilon_min/self.epsilon
        self.epsilon_decay = self.epsilon_decay**(1./float(episodes))

        # Q-network weights filename 
        self.weights_file = 'dqn_cartpole.h5'
        # Q-network for training 
        n_inputs = state_space.shape[0]
        n_outputs = action_space.n 
        self.q_model = self.build_model(n_inputs,n_outputs)
        self.q_model.compile(loss='mse',optimizer =Adam())
        # target Q network 
        self.target_q_model = self.build_model(n_inputs,n_outputs)
        # copy Q network params to target Q network 
        self.update_weights()

        self.replay_counter = 0 
        self.ddqn = True if args.ddqn else False 
    
    def build_model(self,n_inputs,n_outputs):
        '''
        Q-network is 256-256-256 MLP 
        Arguments:
            n_inputs(int)       input dim
            n_outputs(int)      output dim
        Return:
            q_model(Modle)      DQN 
        '''
        inputs = Input(shape=(n_inputs,),name='state')
        x = Dense(256,activation='relu')(inputs)
        x = Dense(256,activation='relu')(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(n_outputs,activation='linear',name='action')(x)

        q_model = Model(inputs,x)
        q_model.summary()
        return q_model
    def act(self,state):
        '''
        eps-greedy policy
        Return  
            action(tensor)      action to execute
        '''
        if np.random.rand()<self.epsilon:
            # explore - do random action 
            return self.action_space.sample()

        # exploit 
        q_values = self.q_model.predict(state)
        # select the action with max-q-value 
        action = np.argmax(q_values[0])
        return action 
    def remember(self,state,action,reward,next_state,done):
        '''
        store experiences in the replay buffer 
        Arguments:
            state(tensor)       env state 
            action(tensor)      agent action
            reward(tensor)      reward received after executeing action on state 
            next_state(tensor)  next state 
        '''
        item = (state,action,reward,next_state,done)
        self.memory.append(item)

    def get_target_q_value(self,next_state,reward):
        '''
        Compute Q_max 
            Use of target Q network solves non-stationarity problem
        
        '''
        # 20230517