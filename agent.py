import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Agent:

    def __init__(self, nr_actions):
        initializer = tf.keras.initializers.GlorotNormal(seed=7)
        inputs = keras.layers.Input(shape=(2))
        x = layers.Dense(5, activation='tanh', kernel_initializer=initializer)(inputs)
        output = layers.Dense(nr_actions, activation='softmax', kernel_initializer=initializer)(x)
        self.policy = keras.Model(inputs=inputs, outputs=output)
        self.policy.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

    def action_based_on_policy(self, state):
        '''
        Returns the chosen action id using the policy for the given state
        '''
        probabilities = self.policy.predict(state)[0]
        nr_actions = len(probabilities)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        return chosen_act

    
