"""
The main model file. Here all the different functions will be pieced together to form the multilayer Neural Network.
If the size of the NN is desired to be changed, it can be done in the layer_dims array.
"""

import numpy as np
from matplotlib import pyplot as plt
from src.prep_data import train_data, test_data, y
from src.initialize_parameters import initialize_parameters
from src.compute_cost import compute_cost
from src.update_parameters import update_parameters
from src.linear_relu_forward import L_model_forward
from src.linear_relu_backward import L_model_backward


plt.rcParams['figure.figsize'] = (15.0, 15.0)  # set default size of plots

layers = [784, 25, 7, 5, 10]  # four layer model, 4th layer having 10 output units which will be rounded off and highest
# probability will be the predicted digit

layer_dims = np.array(layers)  # default dimensions of layers


class VanillaNN:
    """
    The model object
    This will have the train and test functions
    """

    def __init__(self, layer_dimensions=layer_dims, iterations=3000, learning_rate=0.0075, print_cost=False):
        """
        initiating the model object
        :param layer_dimensions: dimensions of each layer (and number of layers) set by the user
        """

        self.layer_dims = layer_dimensions  # user can be allowed to pass in the neural network architecture
        self.parameters = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost

        self.costs = []

    def train(self, X=train_data, Y=y):
        """
        this is the most important function. Here, all the helper functions will be called and model will be trained
        :param Y: label values of the images
        :param X: set of all the images (pixel arrays), in order to train this supervised model
        :return:
        """

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
            if i % 100 == 0:
                print(f"Cost for iteration # {i}:  {cost}")

    def test(self, X_test=test_data):
        """
        computing the predicted digit of an image pixel array
        :param X_test: image pixel array
        :return: int of the predicted digit
        """

        AL, _ = L_model_forward(X_test, self.parameters)

        digit = np.where(AL == np.amax(AL))

        fig = np.asarray(AL).reshape(28, 28)
        plt.title(f"The test example digit is: {digit}")
        plt.imshow(fig)
        plt.show()











































































