import numpy as np
import random
from agent import Agent
from environment import Environment
from utilfunctions import find_initial_state
from utilfunctions import initializer
from utilfunctions import one_hot
from utilfunctions import update_state_step
from rl_utils import monitoring_performance

# the size of the system
SYSTEM_SIZE = 10

# creating the environment
env = Environment(SYSTEM_SIZE)

# creating the agent with 4 actions
# in case you want to change this it should be also changed in the Environment class
nr_actions = 4
agent = Agent(nr_actions=nr_actions, gamma=0.95)

# number of trainings
ROUNDS_OF_TRAINING = SYSTEM_SIZE * SYSTEM_SIZE * 50


# setting the random seeds
random.seed(1)
np.random.seed(2)
training_log = np.array([])
agent.policy.save('./training-results/not-trained-agent-system-size-'+str(SYSTEM_SIZE))

for training_id in range(ROUNDS_OF_TRAINING):

    print("\nround: "+str(training_id))

    # finding a initial state which is not the terminal_state
    initial_state = find_initial_state(env)
    print("...    initial state is "+str(initial_state))

    state, histories, terminated, steps = initializer(initial_state[0, 0], initial_state[0, 1])

    while not terminated:

        action_id = agent.action_based_on_policy(state)
        one_hot_action = one_hot(action_id, nr_actions)
        
        new_state, reward, terminated = env.step(action_id, state)

        histories.appending(reward, state, one_hot_action)

        state, steps = update_state_step(new_state, steps)

        if steps % 100 == 0:
            print("         step "+str(steps)+"  state is "+str(state))
            
    print("...    terminated at: "+str(steps))

    agent.learning(histories)
    
    training_log = monitoring_performance(training_log, training_id, steps, initial_state, env, write_to_disk=True)

    print("...    saving the model")
    agent.policy.save('./training-results/trained-agent-system-size-'+str(SYSTEM_SIZE), save_format='tf')

    print("round: "+str(training_id)+" is finished.")
