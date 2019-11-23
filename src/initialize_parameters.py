"""
This file will initialize all the required parameters
The weights, W, and the biases, b, will be generated for the neural network's layers
"""

import numpy as np

parameters = {}

def initialize_parameters(layers):
    # this function will be imported in model to initialize paramters
    # returns parameters

    L = len(layers)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers[i], layers[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layers[i], 1))

    return parameters
