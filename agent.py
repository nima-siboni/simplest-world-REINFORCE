import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utilfunctions import reshaping_and_reward_to_go_calculations, scale_state

class Agent:
    '''
    the agent class which has the policy.
    - takes actions based on the policy
    - 
    '''
    def __init__(self, nr_actions, gamma=0.99, epsilon=0.02):
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=1)
        optimzer = tf.keras.optimizers.Adam(learning_rate=0.01)
        inputs = keras.layers.Input(shape=(2))
        x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(inputs)
        x = layers.Dense(5, activation='relu', kernel_initializer=initializer)(x)
        output = layers.Dense(nr_actions, activation='softmax', kernel_initializer=initializer)(x)
        output = (output + epsilon) / (1.0 + epsilon * nr_actions)

        self.policy = keras.Model(inputs=inputs, outputs=output)
        self.policy.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
        self.gamma = gamma
        
    def action_based_on_policy(self, state, env):
        '''
        Returns the chosen action id using the policy for the given state
        '''
        scaled_state = scale_state(state, env)
        probabilities = self.policy.predict(scaled_state)[0]
        #print(state, probabilities)
        nr_actions = len(probabilities)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        return chosen_act

    def learning(self, histories):
        '''
        the learning happens here
        '''
        print("...    reshaping the data, and calculating reward to go")
        reward_weighted_actions, scaled_state_history = reshaping_and_reward_to_go_calculations(histories, self.gamma)
#        import pdb; pdb.set_trace()
        print("...    training")
        print("...        with sample size of ", np.shape(scaled_state_history)[0])
        
        fitting_log = self.policy.fit(x=scaled_state_history, y=reward_weighted_actions, epochs=1, verbose=0)
