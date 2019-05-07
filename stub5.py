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
        # 1
        tree_dist_x = state['tree']['dist']
        if tree_dist_x > 400:
            tree_dist_x = 400
        elif tree_dist_x < 0:
            tree_dist_x = 0
        else:
            tree_dist_x = int(tree_dist_x/10)
        # 2
        tree_top_y = int((state['tree']['top'] - state['monkey']['top']) / 5)
        # 3
        tree_bottom_y = int((state['monkey']['bot'] - state['tree']['bot']) / 5)
        # 4
        vel = int(state['monkey']['vel'] / 5)
        d = 1 if vel >= 0 else -1
        # print(vel)
        return (tree_dist_x, tree_top_y, tree_bottom_y, vel)

    def update_q_table(self, state, last_states, last_action, last_reward):
        if len(last_states) == 0:
            return
        last_state = self.reduce_state(last_states[-1])
        # print(last_state)
        cur_state_q = self.states[state]
        n = 0.8
        gamma = 0.2
        self.states[last_state][last_action] = self.states[last_state][last_action] + n * (last_reward - self.states[last_state][last_action] + gamma * max(cur_state_q))

    def __init__(self):
        self.last_states = deque(maxlen=10)
        self.last_action = None
        self.last_reward = None
        self.states = defaultdict(lambda : [0, 0])
        self.misses = 0
        self.hits = 0
        self.highscore = 0

    def reset(self):
        # print('ypppp')
        # print('score', self.last_states[-1]['score'])
        score = self.last_states[-1]['score']
        if score > self.highscore:
            self.highscore = score
        print('High Score', self.highscore)
        self.last_states  = deque(maxlen=10)
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # Reduce the state to just the state we care about
        reduced_state = self.reduce_state(state)

        # Update previous action/reward
        self.update_q_table(reduced_state, self.last_states, self.last_action, self.last_reward)

        # Get new action from policy
        q_table_row = self.states[reduced_state]

        if q_table_row[0] == q_table_row[1]:
            # If there is a tie, guess randomly
            new_action = npr.rand() < 0.1
            self.misses += 1
        else:
            # Else choose action with highest expected reward
            new_action = q_table_row[1] > q_table_row[0]
            self.hits += 1

        # print("percent misses {}".format(self.misses/(self.misses + self.hits)))
        # print(state)
        self.last_action = new_action  
        self.last_states.append(state)
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if reward:
            self.last_reward  = reward * 1000
            return
        # return (tree_dist_x, tree_top_y, tree_bottom_y, vel)
        reduced = self.reduce_state(self.last_states[-1])
        # print(reduced)
        last_state = self.last_states[-1]
        monkey_avg = (last_state['monkey']['top'] + last_state['monkey']['bot']) / 2
        tree_avg = (last_state['tree']['top'] + last_state['tree']['bot']) / 2
        top_dist = abs(last_state['monkey']['top'] - last_state['tree']['top'])
        bot_dist = abs(last_state['monkey']['bot'] - last_state['tree']['bot'])
        penalty = 0
        if last_state['monkey']['top'] + 5 > last_state['tree']['top']: 
            penalty = -1 * top_dist
        elif last_state['monkey']['bot'] - 5 < last_state['tree']['bot']:
            penalty = -1 * bot_dist
        # # vel = last_state['monkey']['vel']
        # # print(vel)
        self.last_reward = (400 - last_state['tree']['dist'])(last_state['monkey']['bot'] - last_state['tree']['top'])
        # self.last_reward =  (800 - 2 *  max([top_dist, bot_dist])) + penalty
# (400 - abs(monkey_avg - tree_avg)) * 0.5 + (50 - abs(reduced[3])) + 

def run_games(learner, hist, iters = 100, t_len = 100):
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

	# Run games. 
	run_games(agent, hist, 150, 10)

	# Save history. 
	np.save('hist',np.array(hist))