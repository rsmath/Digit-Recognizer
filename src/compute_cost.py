"""
This module will compute the cost of each iteration of forward prop
"""


import numpy as np


from src.prep_data import labels

y = []

for i in range(len(labels)):
    temp = np.zeros((10,), dtype=int)
    temp[labels[i]] = 1
    y.append(temp)

y = np.asarray(y) # shape (42000, 10), a 1 for each label digit's position in an empty (10,) zeros array

def compute_cost(AL):
    pass



