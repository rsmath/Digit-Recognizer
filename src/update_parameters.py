"""
This module will update the parameters for gradient descent
"""


def update_parameters(parameters, gradients, alpha=0.01):
    """
    subtracting the gradient from the parameters for gradient descent
    :param parameters: containing all the parameters for all the layers
    :param gradients: containing all the gradients for all the layers
    :param alpha: learning rate
    :return:
    """

    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= alpha * gradients[l - 1][0]
        parameters['b' + str(l)] -= alpha * gradients[l - 1][1]

    return parameters


