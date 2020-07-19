import numpy as np


def calulate_reward_to_go(reward_history):

    reward_to_go = np.zeros_like(reward_history)
    reward_to_go = np.cumsum(reward_history[::-1])
    mean_r_t_g = np.mean(reward_to_go)
    reward_to_go = reward_to_go - mean_r_t_g
    reward_to_go = reward_to_go.reshape(-1, 1)
    reward_to_go = reward_to_go[::-1]
    return reward_to_go
