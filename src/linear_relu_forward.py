"""
This file will implement the forward propagation of the model
Separate functions will be defined so that any form of input will work
"""


import numpy as np
from src.equations import softmax, relu


def forward(A_prev, W, b):
    """
    simply computing the value of z
    :param A_prev: activations of previous layer
    :param W: parameters for this layer
    :param b: bias for this layer
    :return: return value of z and cache of linear parameters, A_prev, W, b to be used in backpropagation
    """

    Z = np.dot(W, A_prev) + b

    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    # linear cache
    cache = (A_prev, W, b) # will be used later in backpropagation

    return Z, cache

def linear_forward(A_prev, W, b, func):
    """
    computing the activations based on the activation function
    :param A_prev: activations of previous layer
    :param W: parameters for this layer
    :param b: bias for this layer
    :param func: relu or softmax
    :return: activated values A plus linear cache and activated cache (z values)
    """

    linear_cache, activated_cache, A = None, None, None

    if func == "softmax":
        Z, linear_cache = forward(A_prev, W, b)
        A, activated_cache = softmax(Z)

    elif func == "relu":
        Z, linear_cache = forward(A_prev, W, b)
        A, activated_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activated_cache)

    return A, cache

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

    for l in range(1, L):
        A_prev = A
        A, cache = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache) # adding layer l's linear and activated caches to be used in backpropagation

    AL, cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax')
    caches.append(cache)

    assert (AL.shape == (10, X.shape[1]))

    return AL, caches


