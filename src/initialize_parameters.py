"""
This file will initialize all the required parameters
The weights, W, and the biases, b, will be generated for the neural network's layers
"""


import numpy as np


parameters = {}

def initialize_parameters(layers):
    """
    initializes the parameters for the layers passed
    :param layers: a list of layer dimensions
    :return: dictionary containing different parameters, W1, W2, b1, b2, etc.
    """

    L = len(layers)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers[i], layers[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layers[i], 1))

        assert (parameters['W' + str(i)].shape == (layers[i], layers[i - 1]))
        assert (parameters['b' + str(i)].shape == (layers[i], 1))

    return parameters


