"""
This file will implement the backward propagation of the model
Separate functions will be defined so that any form of input will work
"""


import numpy as np
from src.equations import sigmoid_backward, relu_backward
from src.prep_data import m, y


def backward(dZ, caches):
    """
    computing the gradients of cost with respect to W, b and A, aka dW, db, and dA[L-1]
    :param dZ: dZ of current layer
    :param caches: packed tuple (linear_cache, activated cache)
    :return: dW, db, and dA_prev
    """

    linear_cache, _ = caches
    W, b, A_prev = linear_cache

    dW = (1 / m) * np.dot(dZ, np.transpose(A_prev))
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    return (dW, db), dA_prev

def linear_backward(dA, caches, func):
    """
    computing the entire backward prop gradients based on the activation function
    :param dA: gradient of cost function with respect to current layer's activations
    :param caches: packed tuple (linear_cache, activated cache)
    :param func: activation function (relu or sigmoid)
    :return: dW, db, dA_prev
    """

    gradient, dA_prev = None, None
    _, activated_cache = caches
    Z = activated_cache

    if func == 'sigmoid':
        dZ = sigmoid_backward(dA, Z) # implementation for sigmoid_backward handles the dZ calculation
        gradient, dA_prev = backward(dZ, caches)

    elif func == 'relu':
        dZ = relu_backward(dA, Z) # implementation for relu_backward handles the dZ calculation
        gradient = backward(dZ, caches)

    return gradient, dA_prev

def L_model_backward(AL, caches):
    """
    computing the gradients for all the layers
    :param AL: output value of final layer after sigmoid activation
    :param caches: accumulated caches of all the layers
    :return: accumulated gradients of all the layers to be used by update_parameters
    """

    gradients = []
    L = len(caches)
    dA = (-y / AL) + ((1 - y) / (1 - AL)) # dA[L] of final layer

    gradient, dA_prev = linear_backward(dA, caches[L], 'relu')
    gradients.append(gradient)

    dA = dA_prev

    for l in range(L - 1, 1, -1):
        gradient, dA_prev = linear_backward(dA, caches[l], 'sigmoid')
        gradients.append(gradient)
        dA = dA_prev

    return gradients

