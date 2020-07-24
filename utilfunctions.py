import numpy as np
from rl_utils import calculate_reward_to_go
from rl_utils import Histories


def one_hot(chosen_act, nr_actions):
    '''
    turn the chosen action to a one hotted vector
    '''
    tmp = np.zeros((1, nr_actions))
    tmp[0, chosen_act] = 1
    return tmp


def initializer(x0=0, y0=0):
    '''
    returns:
    - the initial state, 
    - a bunch of empty arrays,
    - set the terminated to False
    - reset the steps to 0
    '''

    state = np.array([[x0, y0]])
    histories = Histories()
    terminated = False
    steps = 0
    return state, histories, terminated, steps


def shape_adopter(history, m):
    '''
    convert a (x, 1, y)
    '''
    history = np.array(history)
#    import pdb; pdb.set_trace()
#    _, _, m = np.shape(history)
    history = history.reshape(-1, m)

    return history


def reshaping_and_reward_to_go_calculations(histories, gamma):
    '''
    Capsulating the shape_adopter and calculate_reward_to_go
    '''
    state_history = shape_adopter(histories.state_history, 2)
    action_history = shape_adopter(histories.action_history, 4)
    reward_history = shape_adopter(histories.reward_history, 1)

    reward_to_go = calculate_reward_to_go(reward_history, gamma)

    reward_weighted_actions = np.multiply(action_history, reward_to_go)

    return reward_weighted_actions, state_history


def find_initial_state(env):
    '''
    Finding an initial state which is not the final state
    '''
    initial_state = env.TERMINAL_STATE + 0
    while np.array_equal(initial_state, env.TERMINAL_STATE):
        initial_state = np.random.randint(low=0, high=env.SYSTEM_SIZE, size=(1, 2))

    return initial_state

def update_state_step(new_state, step):
    state = new_state + 0
    step = step + 1
    return state, step
    



