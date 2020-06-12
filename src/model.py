"""
The main model file. Here all the different functions will be pieced together to form the multilayer Neural Network.
If the size of the NN is desired to be changed, it can be done in the layer_dims array.
"""

import numpy as np
from matplotlib import pyplot as plt
from src.prep_data import train_data, test_data, cv_data, y, y_cv, labels_train, m_train
from src.initialize_parameters import initialize_parameters
from src.compute_cost import compute_cost
from src.update_parameters import update_adam_parameters, initialize_adam, update_gd_parameters
from src.linear_relu_forward import L_model_forward
from src.linear_relu_backward import L_model_backward


plt.rcParams['figure.figsize'] = (10.0, 7.0)  # set default size of plots

layer_dimensions = [784, 30, 30, 10]  # 3 layer model, 3rd layer having 10 output units which will be rounded off and


def test_accuracy(predictions, ground_truth=None, size=None):
    """
    calculates the accuracy of the predictions
    :param size: size can be passed in, either for training or testing sets
    :param ground_truth: if required, user can pass the ground truth
    :param predictions: digit predictions for each example
    :return: accuracy as a percentage over 100%
    """

    if ground_truth is None:
        ground_truth = labels_train  # default is over the training set

    if size is None:
        size = m_train  # default is for training set

    accuracy = round(np.sum(predictions == ground_truth) * 100 / size, 2)

    return accuracy


def make_batches(X_data, y_data, batch_size=512):
    """
    returns a list of batches of size passed in
    :param X_data: x data passed in
    :param y_data: y data passed in
    :param batch_size: batch size
    :return: list of batches
    """

    total = X_data.shape[1] # number of total examples

    permutation = np.random.permutation(total) # needs to be same for both x and y

    shuffled_x = X_data[:, permutation]
    shuffled_y = y_data[:, permutation].reshape(10, total)

    whole_batches = total // batch_size  # considering data's second dimension contains all examples
    batches = []

    for i in range(whole_batches):
        curr_x = shuffled_x[:, i * batch_size: (i + 1) * batch_size]
        curr_y = shuffled_y[:, i * batch_size: (i + 1) * batch_size]
        batch = (curr_x, curr_y)
        batches.append(batch)

    if total % 2 != 0:
        curr_x = shuffled_x[:, whole_batches * batch_size:]
        curr_y = shuffled_y[:, whole_batches * batch_size:]
        batch = (curr_x, curr_y)
        batches.append(batch)

    return batches


class VanillaNN:
    """
    The model object
    This will have the train and test functions
    """

    def __init__(self, parameters=None, layer_dims=None, iterations=3000, learning_rate=0.075, mini_batch_size=512,
                 print_cost=False):
        """
        initiating the model object
        :param parameters: if passed by user (saw while testing)
        :param mini_batch_size: mini batch size passed in by the user
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
        self.mini_batch_size = mini_batch_size
        self.print_cost = print_cost
        self.costs = []
        self.cv_costs = []
        self.v, self.s = None, None

    def train(self, X=train_data, Y=y, technique="adam"):
        """
        this is the most important function. Here, all the helper functions will be called and model will be trained
        :param technique: optimization algorithm (adam or gd)
        :param Y: label values of the images
        :param X: set of all the images (pixel arrays), in order to train this supervised model
        :return: None
        """

        if technique == 'gd':
            self.iterations = 50

        # first the parameters need to be initialized
        self.parameters = initialize_parameters(self.layer_dims)
        self.v, self.s = initialize_adam(self.parameters)

        t = 0 # initializing for adam

        # now for each cycle of iterations
        for i in range(1, self.iterations + 1):

            # make new batches
            batches = make_batches(X, Y, batch_size=self.mini_batch_size) # for training data
            curr_cost, curr_cv_cost = 0, 0

            for batch in batches:

                curr_X, curr_Y = batch

                # forward propagation run
                AL, caches = L_model_forward(curr_X, self.parameters)
                cv_AL, _ = L_model_forward(cv_data, self.parameters)  # validation test

                # cost is stored
                cost = compute_cost(AL, curr_Y)
                curr_cost += cost
                cv_cost = compute_cost(cv_AL, y_cv)
                curr_cv_cost += cv_cost

                # back propagation will be run
                gradients = L_model_backward(AL, caches, Y_param=curr_Y)  # gradients

                # parameters will be updated
                if technique == 'adam':
                    t += 1
                    self.parameters, self.v, self.s = update_adam_parameters(self.parameters, gradients, self.v,
                                                                             self.s, t, self.learning_rate)

                elif technique == 'gd':
                    self.parameters = update_gd_parameters(self.parameters, gradients, self.learning_rate)

            if technique == 'adam':
                t = 0 # resetting the adam counter

                curr_cost = curr_cost / m_train # average cost
                curr_cv_cost = curr_cv_cost / len(y_cv) # average cost

                self.costs.append(curr_cost)
                self.cv_costs.append(curr_cv_cost)

                # printing the cost every iteration
                if (i == 0 or i % 1 == 0) and self.print_cost:
                    print(f"Cost for iteration # {i}:  {curr_cost}")

            elif technique == 'gd':
                t = 0  # resetting the adam counter

                curr_cost = curr_cost / m_train  # average cost
                curr_cv_cost = curr_cv_cost / len(y_cv)  # average cost

                self.costs.append(curr_cost)
                self.cv_costs.append(curr_cv_cost)

                # printing the cost every 10 iterations
                if (i == 0 or i % 10 == 0) and self.print_cost:
                    print(f"Cost for iteration # {i}:  {curr_cost}")

        return self.parameters, self.costs, self.cv_costs

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
