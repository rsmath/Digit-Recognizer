"""
This file will implement the backward propagation of the model
Separate functions will be defined so that any form of input will work
"""


import numpy as np
from src.equations import softmax_backward, relu_backward
from src.prep_data import m_train, y


def backward(dZ, caches):
    """
    computing the gradients of cost with respect to W, b and A, aka dW, db, and dA[L-1]
    :param dZ: dZ of current layer
    :param caches: packed tuple (linear_cache, activated cache)
    :return: dW, db, and dA_prev
    """

    linear_cache, _ = caches
    A_prev, W, b = linear_cache

    dW = (1 / m_train) * np.dot(dZ, np.transpose(A_prev))
    db = (1 / m_train) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_prev.shape == A_prev.shape)

    return (dW, db), dA_prev

def linear_backward(input, caches, func, Y=y):
    """
    computing the entire backward prop gradients based on the activation function
    :param Y: needs to be passed in to avoid mismatch in mini batch gradient descent
    :param input: can be dA (for relu) or AL (for softmax)
    :param caches: packed tuple (linear_cache, activated cache)
    :param func: activation function (relu or softmax)
    :return: dW, db, dA_prev
    """

    gradient, dA_prev = None, None
    _, activated_cache = caches
    Z = activated_cache

    if func == 'softmax':
        dZ = softmax_backward(input, Z, Y) # implementation for softmax_backward handles the dZ calculation
        gradient, dA_prev = backward(dZ, caches)

    elif func == 'relu':
        dZ = relu_backward(input, Z) # implementation for relu_backward handles the dZ calculation
        gradient, dA_prev = backward(dZ, caches)

    return gradient, dA_prev

def L_model_backward(AL, caches, Y_param=y):
    """
    computing the gradients for all the layers
    :param Y_param: needs to be passed in to avoid mismatch in min batch gradient descent
    :param AL: output value of final layer after softmax activation
    :param caches: accumulated caches of all the layers
    :return: accumulated gradients of all the layers to be used by update_parameters
    """

    gradients = []
    L = len(caches) - 1

    gradient, dA_prev = linear_backward(AL, caches[L], 'softmax', Y=Y_param) # for softmax, AL is needed in the backprop
    gradients.append(gradient)

    dA = dA_prev

    for l in range(L - 1, -1, -1):
        gradient, dA_prev = linear_backward(dA, caches[l], 'relu')
        gradients.append(gradient)
        dA = dA_prev

    gradients = gradients[::-1] # going from first layer to last layer

    return gradients


