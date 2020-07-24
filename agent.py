import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utilfunctions import reshaping_and_reward_to_go_calculations

class Agent:
    '''
    the agent class which has the policy.
    - takes actions based on the policy
    - 
    '''
    def __init__(self, nr_actions, gamma=0.95):
        initializer = tf.keras.initializers.RandomNormal(mean=1, stddev=0.001, seed=1)
        inputs = keras.layers.Input(shape=(2))
        x = layers.Dense(5, activation='tanh', kernel_initializer=initializer)(inputs)
        output = layers.Dense(nr_actions, activation='softmax', kernel_initializer=initializer)(x)
        self.policy = keras.Model(inputs=inputs, outputs=output)
        self.policy.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
        self.gamma = gamma
        
    def action_based_on_policy(self, state):
        '''
        Returns the chosen action id using the policy for the given state
        '''
        probabilities = self.policy.predict(state)[0]
        nr_actions = len(probabilities)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        return chosen_act

    def learning(self, histories):
        '''
        the learning happens here
        '''
        print("...    reshaping the data, and calculating reward to go")
        reward_weighted_actions, state_history = reshaping_and_reward_to_go_calculations(histories, self.gamma)

        print("...    training")
        fitting_log = self.policy.fit(x=state_history, y=reward_weighted_actions, epochs=2, verbose=0)
