"""
The main model file. Here all the different functions will be pieced together to form the multilayer Neural Network.
If the size of the NN is desired to be changed, it can be done in the layer_dims array.
"""

import numpy as np
from matplotlib import pyplot as plt


plt.rcParams['figure.figsize'] = (15.0, 15.0)  # set default size of plots

layers = [784, 25, 7, 5, 10]  # four layer model, 4th layer having 10 output units which will be rounded off and highest
# probability will be the predicted digit

layer_dims = np.array(layers)  # default dimensions of layers


class VanillaNN:
    """
    The model object
    This will have the train and test functions
    """

    def __init__(self, layer_dimensions=layer_dims):
        """
        initiating the model object
        :param layer_dimensions: dimensions of each layer (and number of layers) set by the user
        """

        self.layer_dims = layer_dimensions  # user can be allowed to pass in the neural network architecture
        self.weights = None

    def train(self, X, y):
        """
        this is the most important function. Here, all the helper functions will be called and model will be trained
        :param X: set of all the images (pixel arrays), in order to train this supervised model
        :return:
        """
