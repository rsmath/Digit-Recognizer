"""
This module will compute the cost of each iteration of forward prop
"""


import numpy as np
from src.prep_data import labels


m = len(labels) # m
y = []
for i in range(m):
    temp = np.zeros((10,), dtype=int)
    temp[labels[i]] = 1
    y.append(temp)

y = np.asarray(y) # shape (42000, 10), a 1 for each label digit's position in an empty (10,) zeros array

def compute_cost(AL):
    """
    computing the cost of the loss function given an iteration's output value
    :param AL: shape (42000, 10), each example's prediction being an array of 10 rounded values
    :return: cost of the function L
    """

    cost = (-1 / m) * np.sum(np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))

    cost = np.squeeze(cost) # turns [[17]] into 17

    return cost

