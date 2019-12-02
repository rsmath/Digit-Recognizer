"""
This module will compute the cost of each iteration of forward prop
"""


import numpy as np
from src.prep_data import m_train


def compute_cost(AL, y):
    """
    computing the cost of the loss function given an iteration's output value
    :param y: label values of the images
    :param AL: shape (42000, 10), each example's prediction being an array of 10 rounded values
    :return: cost of the function L
    """

    cost = (-1 / m_train) * np.sum(y * np.log(AL + 1e-8)) # 1e-8 added to avoid taking log of 0
    cost = np.squeeze(cost) # turns [[17]] into 17

    assert (cost.shape == ())

    return cost


