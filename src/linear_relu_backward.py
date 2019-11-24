"""
This file will implement the backward propagation of the model
Separate functions will be defined so that any form of input will work
"""


import numpy as np
from src.equations import sigmoid_backward, relu_backward
from src.compute_cost import m


def backward(dA, dZ, caches):
    """
    computing the gradients of cost with respect to W, b and A, aka dW, db, and dA[L-1]
    :param dA: gradient of cost function with respect to current layer's activations
    :param dZ: dZ of current layer
    :param caches: packed tuple (linear_cache, activated cache)
    :return: dW, db, and dA_prev
    """

    linear_cache, activated_cache = caches
    W, b, A_prev = linear_cache
    Z = activated_cache

    dW = (1 / m) * np.dot(dZ, np.transpose(A_prev))
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    return dW, db, dA_prev

def linear_backward(dA, caches, func):
    """
    computing the entire backward prop gradients based on the activation function
    :param dA: gradient of cost function with respect to current layer's activations
    :param dZ: dZ of current layer
    :param caches: packed tuple (linear_cache, activated cache)
    :param func: activation function (relu or sigmoid)
    :return: dW, db, dA_prev
    """

    gradients = {}
    _, activated_cache = caches
    Z = activated_cache

    if func == 'sigmoid':
        dZ = sigmoid_backward(dA, Z) # implementation for sigmoid_backward handles the dZ calculation
        gradient = backward(dA, dZ, caches)


def L_model_forward(X, parameters):
    """
    Computing the entire forward propagation for the model
    :param X: A_prev[0] passed in
    :param parameters: containing all the initialized (or trained) parameters
    :return: output value and the accumulated caches
    """

    A = X # X is inserted as A_prev into the loop
    caches = []
    L = len(parameters) // 2 # for every layer there are W and b, so half of parameters are total number of layers

    for l in range(1, L - 1):
        A_prev = A

        A, cache = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')

        caches.append(cache) # adding layer l's linear and activated caches to be used in backpropagation

    AL, cache = linear_forward(A, parameters['W' + str(L)], parameters['W' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches

