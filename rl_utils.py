import numpy as np


def calculate_reward_to_go(reward_history, gamma):
    '''
    calculating the reward to go hat{Q} for each state in the history,
    where hat{Q} is all the reward you get from on state into the future.
    
    Key arguments:
    reward_history -- the rewards for all the visited states
    gamma -- the discount factor

    output:
    the hat{Q} which is of the same shape as reward_history.
    '''

    reward_to_go = (np.zeros_like(reward_history)).astype(float)
    n = len(reward_history)
    tmp = np.array([np.power(gamma, x+0.0) for x in range(n)])
    tmp = tmp.reshape(-1, 1)
    for i in range(n):
        reward_to_go[i] = ((reward_history.T).dot(tmp))[0]#Uuuugly
        tmp = np.roll(tmp, 1)
        tmp[0] = 0

    mean_r_t_g = np.mean(reward_to_go)
    reward_to_go = reward_to_go - mean_r_t_g
    reward_to_go = reward_to_go.reshape(-1, 1)
    return reward_to_go


def monitoring_performance(log, training_id, steps, initial_state, env, write_to_disk=True):
    '''
    returns a log (a numpy array) which has some analysis of the each round of training.

    Key arguments:

    training_id -- the id of the iteration which is just finished.
    steps -- the number of steps for the agent to reach to the terminal state in this iteration
    initial_state -- the initial state of the agent in this iteration
    env -- the environment (only the terminal_state is extracted from it)
    write_to_disk -- a flag for writting the performance to the disk

    Output:

    a numpy array with info about the iterations and the learning
    '''

    steps_for_the_optimal_policy = np.sum(env.TERMINAL_STATE - initial_state)

    assert steps_for_the_optimal_policy > 0

    performance = steps_for_the_optimal_policy / steps

    if training_id == 0:
        log = np.array([[training_id, performance, steps]])
    else:
        log = np.append(log, np.array([[training_id, performance, steps]]), axis=0)

    if write_to_disk:
        np.savetxt('reward_vs_iteration.dat', log)

    return log


class Histories():
    '''
    just a class to hold data
    '''
    def __init__(self):
        self.scaled_state_history = []
        self.reward_history = []
        self.action_history = []

    def appending(self, reward, scaled_state, one_hot_action):
        self.reward_history.append(reward)
        self.scaled_state_history.append(scaled_state)
        self.action_history.append(one_hot_action)
