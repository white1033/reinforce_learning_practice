'''
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
'''

import time
import sys

import numpy as np
import pandas as pd

N_STATES = 6  # length of one dim array
ACTIONS = ('left', 'right')
EPSILON = 0.9  # greedy police
ALPHA = 0.1
LAMBDA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3


def build_qtable(n_states, actions):
    '''
    return q table for q-learning
    '''
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions, )

    return table


def choose_action(state, q_table):
    '''choose an action from given state'''
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        next_action = np.random.choice(ACTIONS)
    else:
        next_action = state_actions.argmax()
    return next_action


def get_env_feedback(state, action):
    '''
    interact with environment
    '''
    if action == 'right':
        if state == N_STATES - 2:
            next_state, reward = 'terminal', 1
        else:
            next_state, reward = state + 1, 0
    else:
        next_state, reward = (state, 0) if state == 0 else (state - 1, 0)
    return next_state, reward


def update_env(state, episode, step_counter):
    '''
    update current env
    '''
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1,
                                                        step_counter)
        sys.stdout.write('\r{}'.format(interaction))
        sys.stdout.flush()
        time.sleep(2)
        sys.stdout.write('\r                                ')
        sys.stdout.flush()
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        sys.stdout.write('\r{}'.format(interaction))
        sys.stdout.flush()
        time.sleep(FRESH_TIME)


def main():
    '''
    main part of RL loop
    '''
    q_table = build_qtable(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)
        while not is_terminated:

            action = choose_action(state, q_table)
            next_state, reward = get_env_feedback(state, action)
            q_predict = q_table.ix[state, action]
            if next_state != 'terminal':
                q_target = reward + LAMBDA * q_table.iloc[next_state, :].max()
            else:
                q_target = reward  # next state is terminal
                is_terminated = True  # terminate this episode

            q_table.ix[state, action] += ALPHA * \
                (q_target - q_predict)  # update
            state = next_state  # move to next state

            update_env(state, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = main()
    print '\r\nQ-table:'
    print q_table
