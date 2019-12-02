"""
This file will contain the different mathematical functions that the model will require
These functions are: softmax, relu, softmax_backward, relu_backward
"""


import numpy as np


def softmax(Z):
    """
    computing softmax of Z
    :param Z: input
    :return: softmax and cache (z)
    """

    e_Z = np.exp(Z - np.max(Z))

    A = e_Z / e_Z.sum(axis=0)  # only difference

    assert (A.shape == Z.shape)

    cache = Z

    return A, cache


def softmax_backward(AL, cache, y):
    """
    computes dZ for softmax
    :param cache: Z value
    :param AL: output layer A value
    :param y: ground truth
    :return: dZ
    """

    Z = cache

    dZ = AL - y  # derivative of cost with respect to Z for softmax function

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    """
    computing relu of Z
    :param Z: input
    :return: relu and cache (Z)
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z  # cache is used in backprop

    return A, cache


def relu_backward(dA, cache):
    """
    computing dZ
    :param dA: dA of current layer
    :param cache: Z from relu
    :return: dZ
    """

    Z = cache

    dZ = np.array(dA, copy=True)  # gradient is 1 for z > 0

    dZ[Z <= 0] = 0  # gradient is 0 for z <= 0 otherwise 1 for rest

    assert (dZ.shape == Z.shape)

    return dZ


