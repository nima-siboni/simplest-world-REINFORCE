import numpy as np


def calulate_reward_to_go(reward_history, gamma):
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
