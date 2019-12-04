"""
The main model file. Here all the different functions will be pieced together to form the multilayer Neural Network.
If the size of the NN is desired to be changed, it can be done in the layer_dims array.
"""


import numpy as np
from matplotlib import pyplot as plt
from src.prep_data import train_data, test_data, y, labels_train, m_train
from src.initialize_parameters import initialize_parameters
from src.compute_cost import compute_cost
from src.update_parameters import update_parameters
from src.linear_relu_forward import L_model_forward
from src.linear_relu_backward import L_model_backward


plt.rcParams['figure.figsize'] = (10.0, 10.0)  # set default size of plots

layer_dimensions = [784, 10, 5, 10]  # 3 layer model, 3rd layer having 10 output units which will be rounded off and
# highest probability will be the predicted digit


def test_accuracy(predictions, ground_truth=None, size=None):
    """
    calculates the accuracy of the predictions
    :param size: size can be passed in, either for training or testing sets
    :param ground_truth: if required, user can pass the ground truth
    :param predictions: digit predictions for each example
    :return: accuracy as a percentage over 100%
    """

    if ground_truth is None:
        ground_truth = labels_train # default is over the training set

    if size is None:
        size = m_train # default is for training set

    accuracy = round(np.sum(predictions == ground_truth) * 100 / size, 2)

    return accuracy


class VanillaNN:
    """
    The model object
    This will have the train and test functions
    """

    def __init__(self, parameters=None, layer_dims=None, iterations=3000, learning_rate=0.075,
                 print_cost=False):
        """
        initiating the model object
        :param parameters: if passed by user (saw while testing)
        :param layer_dims: network architecture
        :param iterations: gradient descent runs for these many iterations
        :param learning_rate: alpha in gradient descent
        :param print_cost: if required, cost may be printed
        """

        if layer_dims is None:
            self.layer_dims = layer_dimensions

        else:
            self.layer_dims = layer_dims  # user can be allowed to pass in the neural network architecture

        self.parameters = parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.costs = []

    def train(self, X=train_data, Y=y):
        """
        this is the most important function. Here, all the helper functions will be called and model will be trained
        :param Y: label values of the images
        :param X: set of all the images (pixel arrays), in order to train this supervised model
        :return: None
        """

        costs = []

        # first the parameters need to be initialized
        self.parameters = initialize_parameters(self.layer_dims)

        # now for each cycle of iterations
        for i in range(1, self.iterations + 1):

            # forward propagation run
            AL, caches = L_model_forward(X, self.parameters)

            # cost is stored
            cost = compute_cost(AL, Y)
            self.costs.append(cost)

            # back propagation will be run
            gradients = L_model_backward(AL, caches) # gradients

            # parameters/parameters will be updated
            self.parameters = update_parameters(self.parameters, gradients, self.learning_rate)

            # printing the cost every 100 iterations
            if (i == 0 or i % 10 == 0) and self.print_cost:
                costs.append(cost)
                print(f"Cost for iteration # {i}:  {cost}")

        return self.parameters, costs

    def test(self, parameters=None, X_test=test_data):
        """
        computing the predicted digit of an image pixel array
        :param parameters: user inserted
        :param X_test: image pixel array
        :return: output probabilities of softmax
        """

        if parameters is None:
            parameters = self.parameters

        AL, _ = L_model_forward(X_test, parameters)

        return AL


