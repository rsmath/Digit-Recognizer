"""
This module will compute the cost of each iteration of forward prop
"""


import numpy as np


def compute_cost(AL, y):
    """
    computing the cost of the loss function given an iteration's output value
    :param y: label values of the images
    :param AL: shape (42000, 10), each example's prediction being an array of 10 rounded values
    :return: cost of the function L
    """

    cost = (-1 / m) * np.sum(np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
    cost = np.squeeze(cost) # turns [[17]] into 17

    return cost

