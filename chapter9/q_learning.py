'''
A simple deterministic MDP with six states
'''
from collections import deque 
import numpy as np 
import os 
import time 
import argparse
from termcolor import colored

class QWorld:
    def __init__(self):
        '''
        Simulated deterministic world mmade of 6 states 
        Q-learning by Bellman Equation
        '''
        # 4 actions 
        # 0-left 1-down 2-right 3-up
        self.col = 4 

        # 6 states 
        self.row = 6 

        # set up the env 
        self.q_table = np.zeros([self.row,self.col])
        self.init_transition_table()
        self.init_reward_table()

        # discount factor 
        self.gamma = 0.9 

        # 90% exploration 10% exploitation 
        self.epsilon = 0.9 

        # exploration decays by this factor every episode 
        self.epsilon_decacy = 0.9 

        # in the long run 10% exploration 90% exploitation
        self.epsilon_min = 0.1 

        # reset the env 
        self.reset()
        self.is_explore = True
    
    def reset(self):
        '''
        start of episode
        '''
        self.state = 0 
        return self.state
    
    def is_in_win_state(self):
        '''agnet wins when the goal is reached'''
        return self.state == 2
    
    def init_reward_table(self):
        """
        0 - Left, 1 - Down, 2 - Right, 3 - Up
        ----------------
        | 0 | 0 | 100  |
        ----------------
        | 0 | 0 | -100 |
        ----------------
        """
        self.reward_table = np.zeros([self.row,self.col])
        self.reward_table[1,2] = 100
        self.reward_table[4,2] = -100 

    def init_transition_table(self):
        """
        0 - Left, 1 - Down, 2 - Right, 3 - Up -------------
        |0|1|2 |
        -------------
        |3|4|5 |
        -------------
        """
        self.transition_table = np.zeros([self.row,self.col],dtype=int)
        self.transition_table[0, 0] = 0
        self.transition_table[0, 1] = 3
        self.transition_table[0, 2] = 1
        self.transition_table[0, 3] = 0
        self.transition_table[1, 0] = 0
        self.transition_table[1, 1] = 4
        self.transition_table[1, 2] = 2
        self.transition_table[1, 3] = 1
        # terminal Goal state
        self.transition_table[2, 0] = 2
        self.transition_table[2, 1] = 2
        self.transition_table[2, 2] = 2
        self.transition_table[2, 3] = 2
        self.transition_table[3, 0] = 3
        self.transition_table[3, 1] = 3
        self.transition_table[3, 2] = 4
        self.transition_table[3, 3] = 0
        self.transition_table[4, 0] = 3
        self.transition_table[4, 1] = 4
        self.transition_table[4, 2] = 5
        self.transition_table[4, 3] = 1
        # terminal Hole state
        self.transition_table[5, 0] = 5
        self.transition_table[5, 1] = 5
        self.transition_table[5, 2] = 5
        self.transition_table[5, 3] = 5

    def step(self,action):
        '''
        excute the action on the env 
        Argument:
            action(tensor): An action in Action space
        Returns:
            next_state(tensor): next env state 
            reward(float):      reward received by the agent 
            done(bool):         whether the terminal state is reached
        '''
        # determine the next_state given state and actin 
        next_state = self.transition_table[self.state,action]
        # done is True is next_state is goal or hole 
        if next_state==2 :
            done = True
        elif next_state==5:
            done =True
        else:
            done = False 
        # done = next_state == 2 or next_state==5 
        # reward given the state and action
        reward = self.reward_table[self.state,action]
        # the env 
        self.state = next_state
        return next_state,reward,done 
    def act(self):
        '''
        determine the next action either fr Q table (exploitation ) or random (exploration)
        Return:
            action(tensor): action that the agent must execute 
        '''
        # 0-left 1-down 2-right 3-up
        if np.random.rand()<=self.epsilon:
            # explore -do random action 
            self.is_explore = True 
            return np.random.choice(4,1)[0]
        
    def update_q_table(self,state,action,reward,next_state):
        '''
        Q-learning -- update the Q table using Q(s,a)
        Arguments:
            state(tensor):      agent state
            action(tensor):     action executed by the agent 
            reward(float):      reward after executing action for a given state 
            next_state(tensor): next state after executing action for a given state 

        '''
        q_value = self.gamma * np.amax(self.q_table[next_state])
        q_value += reward 
        self.q_table[state,action] = q_value
    def update_epsilon(self):
        '''
        update Exploration-Exploitation mix 
        ''' 
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decacy
    
    def print_world(self,action,step):
        actions = { 0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)" }
        explore = "Explore" if self.is_explore else "Exploit"
        print("Step", step, ":", explore, actions[action])
        for _ in range(13):
            print('-', end='')
        self.print_cell()
        for _ in range(13):
            print('-', end='')
        self.print_cell(row=1)
        for _ in range(13):
            print('-', end='')
        print("")
    def print_cell(self, row=0):
        """UI to display agent moving on the grid"""
        print("")
        for i in range(13):
            j = i - 2
            if j in [0, 4, 8]: 
                if j == 8:
                    if self.state == 2 and row == 0:
                        marker = "\033[4mG\033[0m"
                    elif self.state == 5 and row == 1:
                        marker = "\033[4mH\033[0m"
                    else:
                        marker = 'G' if row == 0 else 'H'
                    color = self.state == 2 and row == 0
                    color = color or (self.state == 5 and row == 1)
                    color = 'red' if color else 'blue'
                    print(colored(marker, color), end='')
                elif self.state in [0, 1, 3, 4]:
                    cell = [(0, 0, 0), (1, 0, 4), (3, 1, 0), (4, 1, 4)]
                    marker = '_' if (self.state, row, j) in cell else ' '
                    print(colored(marker, 'red'), end='')
                else:
                    print(' ', end='')
            elif i % 4 == 0:
                    print('|', end='')
            else:
                print(' ', end='')
        print("")
    def print_q_table(self):
        """UI to dump Q Table contents"""
        print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
        print(self.q_table)


def print_status(q_world, done, step, delay=1):
    """UI to display the world, 
        delay of 1 sec for ease of understanding
    """
    os.system('clear')
    q_world.print_world(action, step)
    q_world.print_q_table()
    if done:
        print("-------EPISODE DONE--------")
        delay *= 2
    time.sleep(delay)
    
def print_episode(episode, delay=1):
    """UI to display episode count
    Arguments:
        episode (int): episode number
        delay (int): sec delay

    """
    os.system('clear')
    for _ in range(13):
        print('=', end='')
    print("")
    print("Episode ", episode)
    for _ in range(13):
        print('=', end='')
    print("")
    time.sleep(delay)


if __name__ =="__main__":
    episode_count = 200 
    q_world = QWorld()
    wins = 0 
    scores = []
    step =  1 
    maxwins = 0 


    for episode in range(episode_count):
        state = q_world.reset()
        done = False
        print_episode(episode)
        while not done:
            action = q_world.act()
            next_state,reward,done = q_world.step(action)
            q_world.update_q_table(state,action,reward,next_state)
            print_status(q_world,done,step)
            state = next_state
            if done:
                if q_world.is_in_win_state():
                    wins +=1 
                    scores.append(step)
                    if wins>maxwins:
                        print(scores)
                        exit(0)

                # exploration-Exploitation is updated every episode 
                q_world.update_epsilon()
                step+=1 
            else:
                step += 1 
                





