"""
This file will contain the different mathematical functions that the model will require
These functions are: sigmoid, relu, sigmoid_backward, relu_backward
"""


import numpy as np


def sigmoid(Z):
    """
    computing sigmoid of Z
    :param Z: input
    :return: sigmoid and cache (z)
    """

    cache = Z # cache is used in backprop

    return 1 / (1 + np.exp(-Z)), cache

def relu(Z):
    """
    computing relu of Z
    :param Z: input
    :return: relu and cache (Z)
    """
    cache = Z # cache is used in backprop

    return np.maximum(0, Z), cache

def sigmoid_backward(dA, cache):
    """
    computing dZ
    :param dA: dA of current layer
    :param cache: Z from sigmoid
    :return: dZ
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))

    dZ = np.multiply(dA, s * (1 - s)) # derivative of cost with respect to Z for sigmoid function

    return dZ

def relu_backward(dA, cache):
    """
    computing dZ
    :param dA: dA of current layer
    :param cache: Z from relu
    :return: dZ
    """

    Z = cache

    dZ = np.ones_like(dA) # need dZ to be z numpy array for next step

    dZ[Z <= 0] = 0 # gradient is 0 for z <= 0 otherwise 1 for rest

    return dZ

