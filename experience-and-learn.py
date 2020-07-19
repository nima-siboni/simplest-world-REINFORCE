import numpy as np
import random
from agent import Agent
from environment import Environment
from rl_utils import calulate_reward_to_go
from utilfunctions import one_hot
from utilfunctions import shape_adopter

SYSTEM_SIZE = 4
env = Environment(SYSTEM_SIZE)
agent = Agent(nr_actions=4)
ROUND_OF_TRAINING = 50

random.seed(1)
np.random.seed(1)

for training_id in range(ROUND_OF_TRAINING):

    print("round: "+str(training_id))
    state = np.array([[0, 0]])
    state_history = []
    reward_history = []
    action_history = []
    env.terminated = False
    steps = 0

    while not env.terminated:

        action_id = agent.action_based_on_policy(state)
        
        new_state, reward = env.step(action_id, state)
        
        reward_history.append(reward)
        state_history.append(state)
        action_history.append(one_hot(action_id, 4))
        
        state = new_state + 0
        steps += 1

    print("...     terminated at: "+str(steps))
    print("...     reshaping the data")

    state_history = shape_adopter(state_history, 2)
    action_history = shape_adopter(action_history, 4)
    reward_history = shape_adopter(reward_history, 1)

    reward_to_go = calulate_reward_to_go(reward_history)

    reward_weighted_actions = np.multiply(action_history, reward_to_go)

    print("...     training")
    training_log = agent.policy.fit(x=state_history, y=reward_weighted_actions, epochs=100, verbose=0)
