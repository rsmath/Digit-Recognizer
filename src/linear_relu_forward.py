"""
This file will implement the forward propagation of the model
Separate functions will be defined so that any form of input will work
"""

import numpy as np
from src.equations import sigmoid, relu

def forward(A_prev, W, b):
    # this function will compute the value of Z and return it

    Z = np.dot(W, A_prev) + b

    # linear cache
    cache = (A_prev, W, b) # will be used later in backprop

    return Z, cache

def linear_forward(A_prev, W, b, func):
    # computing the activation function of the Z value

    if func == "sigmoid":
        Z, linear_cache = forward(A_prev, W, b)
        A_prev, activated_cache = sigmoid(Z)

    elif func == "relu":
        Z, linear_cache = forward(A_prev, W, b)
        A_prev, activated_cache = relu(Z)

    cache = (linear_cache, activated_cache)

    return A_prev, cache

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

        caches.append(cache) # adding layer l's linear and activated caches to be used in backprop

    AL, cache = linear_forward(A, parameters['W' + str(L)], parameters['W' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches




