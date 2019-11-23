"""
This file will implement the forward propagation of the model
Separate functions will be defined so that any form of input will work
"""

import numpy as np
from src.equations import sigmoid, relu

def forward(A, W, b):
    # this function will compute the value of Z and return it

    Z = np.dot(W, A) + b

    cache = (A, W, b) # will be used later in backprop

    return Z, cache

def linear_forward(Z, func):
    # computing the activation function of the Z value

    if func == "sigmoid":
        A = sigmoid(Z)

    elif func == "relu":
        A = relu(Z)

    return A


