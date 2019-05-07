# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import random

from SwingyMonkey import SwingyMonkey
from collections import defaultdict, deque

class Learner(object):
    '''
    This agent jumps randomly.
    '''
    def reduce_state(self, state):
        if not state:
            return None
        
        tree_dist_x = state['tree']['dist']
        if tree_dist_x > 300:
            tree_dist_x = 300
        elif tree_dist_x < 0:
            tree_dist_x = 0
        else:
            tree_dist_x = int(tree_dist_x / 50)

        tree_top_y = int((state['tree']['top'] - state['monkey']['top']) / 50)
        tree_bottom_y = int((state['monkey']['bot'] - state['tree']['bot']) / 50)

        vel = 1 if int(state['monkey']['vel']/5) > 0 else -1
        return (tree_dist_x, tree_top_y, tree_bottom_y, vel, self.gravity)

    def update_q_table(self, state, last_states, last_action, last_reward):
        if len(last_states) == 0:
            return
        last_state = self.reduce_state(last_states[-1])
        cur_state_q = self.states[state]
        self.states[last_state][last_action] = self.states[last_state][last_action] + self.n * (last_reward - self.states[last_state][last_action] + self.gamma * max(cur_state_q))

    def __init__(self):
        self.last_states = deque(maxlen=10)
        self.last_action = None
        self.last_reward = None
        self.states = defaultdict(lambda : [0, 0])
        self.frame = 0
        
        self.iteration = 0
        self.misses = 0
        self.hits = 0
        self.highscore = 0
        self.gravity = 0

        self.n = 0.6
        self.gamma = 0.9
        self.epsilon = 0.2

    def reset(self):
        score = self.last_states[-1]['score']

        if score > self.highscore:
            self.highscore = score

        print('High Score', self.highscore, self.iteration)

        self.iteration += 1

        self.last_states  = deque(maxlen=10)
        self.last_action = None
        self.last_reward = None
        self.frame = 0
        self.gravity = 0

        # self.n = self.n - (0.5 / 100)
        # self.epsilon = self.epsilon - (0.15 / 40)
        self.n = 0.6 if self.iteration < 30 else 0.3
        self.epsilon = 0.2 if self.iteration < 30 else 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # Reduce the state to just the state we care about
        self.frame += 1
        
        if (self.frame == 1):
            self.last_states.append(state)
            self.last_reward = 0
            self.last_action = 0
            return 0

        else:
            self.gravity = abs(state["monkey"]["vel"] - self.last_states[-1]["monkey"]["vel"]) / 2
            reduced_state = self.reduce_state(state)
            # print (reduced_state)
            # Update previous action/reward
            self.update_q_table(reduced_state, self.last_states, self.last_action, self.last_reward)

            # Get new action from policy
            q_table_row = self.states[reduced_state]

            if q_table_row[0] == q_table_row[1]:
                # If there is a tie, guess randomly
                new_action = npr.rand() < 0.01
                self.misses += 1
            else:
                # Else choose action with highest expected reward
                new_action = q_table_row[1] > q_table_row[0]
                self.hits += 1

            if (self.epsilon > npr.random_sample()):
                new_action = npr.rand() < 0.01

            self.last_action = new_action  
            self.last_states.append(state)
            return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if reward:
            self.last_reward  = reward * 1000
            return
            
        last_state = self.last_states[-1]
        monkey_avg = (last_state['monkey']['top'] + last_state['monkey']['bot']) / 2
        tree_avg = (last_state['tree']['top'] + last_state['tree']['bot']) / 2

        self.last_reward = 400 - abs(tree_avg - monkey_avg)
        target_vel_direction = -1 if monkey_avg > tree_avg else 1
        self.last_reward += 100 * (last_state['monkey']['vel']/10) * target_vel_direction

def run_games(learner, hist, iters = 100, t_len = 100):
    print(t_len)
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    print(hist)
    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    for i in range(0, 10):
        # Run games. 
        run_games(agent, hist, 100, 1)

    # Save history. 
    np.save('hist',np.array(hist))