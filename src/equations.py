'''
This file will contain the different mathematical functions that the model will require
These functions are: sigmoid, relu, sigmoid_backward, relu_backward
'''


import numpy as np

def sigmoid(a):
    # returning the sigmoid value of the passed parameter

    cache = a # cache is used in backprop

    return 1 / (1 + np.exp(-a)), cache

def relu(a):
    # returning the relu activation function value of the parameter

    cache = a # cache is used in backprop

    return np.maximum(0, a), cache

def sigmoid_backward(dA, cache):
    # returns the gradient of cost with respect to Z, dZ

    Z = cache

    s = 1 / (1 + np.exp(-Z))

    dZ = dA * s * (1 - s) # derivative of cost with respect to Z for sigmoid function

    return dZ

def relu_backward(dA, cache):
    # returns the gradient of cost with respect to Z, dZ for relu function

    Z = cache

    dZ = np.ones_like(dA) # need dZ to be a numpy array for next step

    dZ[Z <= 0] = 0 # gradient is 0 for z <= 0 otherwise 1 for rest

    return dZ

