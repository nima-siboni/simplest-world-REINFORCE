import numpy as np


def one_hot(chosen_act, nr_actions):
    tmp = np.zeros((1, nr_actions))
    tmp[0, chosen_act] = 1
    return tmp


def shape_adopter(history, m):

    history = np.array(history)
#    import pdb; pdb.set_trace()
#    _, _, m = np.shape(history)
    history = history.reshape(-1, m)

    return history


def calulated_reward_to_go(reward_history):

    reward_to_go = np.zeros_like(reward_history)
    reward_to_go = np.cumsum(reward_history[::-1])
    mean_r_t_g = np.mean(reward_to_go)
    reward_to_go = reward_to_go - mean_r_t_g
    reward_to_go = reward_to_go.reshape(-1, 1)
    reward_to_go = reward_to_go[::-1]
    return reward_to_go
