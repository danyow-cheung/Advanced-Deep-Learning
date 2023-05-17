from collections import deque
import numpy as np
import argparse
import os 
import time 
import gym 
from gym import wrappers,logger

class QAgent:
    def __init__(self,observation_space,action_space,demo=False,slippery=False,episodes=40000):
        '''
        Q-learning agent on FrozenLake-v0 env 
        Arguments:
            observation_space(tensor)       state space
            action_space(tensor)            action space 
            demo(Bool)                      whether for demo or training 
            slippery(bool)                  2 version of FLv0 env
            episodes(int)                   number of episodes to train 
        '''
        self.action_space = action_space
        # number of columns is equal to number of actions 
        col = action_space.n 

        # number of row is equal to number of states 
        row = observation_space.n 

        # build Q table with row x col dims 
        self.q_table = np.zeros([row,col])

        # discount factor 
        self.gamma = 0.9 
        # initially 90% exploration ,10% exceplotation
        self.epsilon = 0.9 
        # interatively applying decay til 10% exploration /90% exploitation
        self.epsilon_min = 0.1 
        self.epsilon_decay = self.epsilon_min/self.epsilon
        self.epsilon_decay = self.epsilon_decay** (1./float(episodes))

        # learning rate of q-learning 
        self.learning_rate = 0.1 

        # file where q table is saved on/restored fr 
        if slippery:
            self.filename = 'q-frozenlake-slippery.npy'
        else:
            self.filename = 'q-frozenlake.npy'

        # demo or train mode 
        self.demo = demo 
        if demo:
            self.epsilon = 0 
    def act(self,state,is_explore=False):
        '''
        detemine the next action
            if random,choose from random action space 
            else use the Q table 
        Arguments:
            state(tensor):      agent's current state
            is_explore(Bool):   exploratin mode or not 
        Return:
            action(tensor):     action that the agent must execute
        '''
        if is_explore or np.random.rand()<self.epsilon:
            return self.action_space.sample()

        action = np.argmax(self.q_table[state])
        return action
    def update_q_table(self,state,action,reward,next_state,):
        '''
        TD(0)-learning (generalizaed Q-learning ) with learning rate
        Arguments:
            state(tensor)       env state
            action(tensor)      action executed by the agent for the given state
            reward(float)       reward received by the agent for executing the action
            next_state(tensor)  the env next state 
        '''
        q_value = self.gamma * np.amax(self.q_table[next_state])
        q_value += reward 
        q_value -= self.q_table[state,action]
        q_value *= self.learning_rate
        q_value += self.q_table[state,action]
        self.q_table[state,action] = q_value

    def update_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id',
                        nargs='?',
                        default='FrozenLake-v0',
                        help='Select the environment to run')
    help_ = "Demo learned Q Table"
    parser.add_argument("-d",
                        "--demo",
                        help=help_,
                        action='store_true')
    help_ = "Frozen lake is slippery"
    parser.add_argument("-s",
                        "--slippery",
                        help=help_,
                        action='store_true')
    help_ = "Exploration only. For baseline."
    parser.add_argument("-e",
                        "--explore",
                        help=help_,
                        action='store_true')
    help_ = "Sec of time delay in UI. Useful for viz in demo mode."
    parser.add_argument("-t",
                        "--delay",
                        help=help_,
                        type=int)
    args = parser.parse_args()

    wins = 0 
    episodes = 2 
    


    logger.setLevel(logger.INFO)

    # instantiate a gym environment (FrozenLake-v0)
    # env = gym.make('FrozenLake-v1')

    env = gym.make(args.env_id)
    # loop for the specified number of episode 
    agent = QAgent(env.observation_space,
                   env.action_space,
                   demo=args.demo,
                   slippery=args.slippery,
                   episodes=episodes)
    
    for episode in range(episodes):
        state = env.reset()
        done = False 
        while not done:
            # detemine the agent's action given state 
            action = agent.act(state,is_explore=True)
            # get observable data 
            next_state,reward,done,_ = env.step(action)
            # clear the screen before rendering the env 
            os.system('clear')
            # render the env for human debuggin 
            env.render()
            # training of Q table 
            if done:
                # update exploration-exploitation ratio 
                if reward>0:
                    wins += 1
            if True:
                agent.update_q_table(state,action,reward,next_state)
                agent.update_epsilon()
            state = next_state
            percent_wins = 100.0*wins/(episode+1)



