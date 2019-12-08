"""
This module will update the parameters for gradient descent (adam or otherwise)
"""


import numpy as np


def initialize_adam(parameters):
    """
    initiating the Adam parameters
    """

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

    return v, s


def update_adam_parameters(parameters, gradients, v, s, t, alpha=0.01):
    """
    updating the parameters with adam optimization
    :param parameters: passed in parameters
    :param gradients: derivatives of parameters with respect to cost
    :param v: momentum estimation velocity
    :param s: RMSProp estimation
    :param t: batch
    :param alpha: learning rate
    :return: parameters, v, and s
    """

    # predefined constants
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * gradients[l][0]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * gradients[l][1]

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(gradients[l][0])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(gradients[l][1])

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (
                (alpha * v_corrected["dW" + str(l + 1)]) / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (
                (alpha * v_corrected["db" + str(l + 1)]) / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    return parameters, v, s


def update_gd_parameters(parameters, gradients, alpha=0.01):
    """
    subtracting the gradient from the parameters for gradient descent
    :param parameters: containing all the parameters for all the layers
    :param gradients: containing all the gradients for all the layers
    :param alpha: learning rate
    :return: parameters
    """

    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= alpha * gradients[l - 1][0]
        parameters['b' + str(l)] -= alpha * gradients[l - 1][1]

    return parameters


