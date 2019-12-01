"""
This module is just to run the model in a clean environment with new data
"""


import pickle
from src.model import VanillaNN, test_accuracy
from src.prep_data import test_data, train_data, m
from matplotlib import pyplot as plt
import numpy as np


parameters, costs = None, None
layers = [784, 30, 30, 10]

model = VanillaNN(layer_dims=layers, iterations=500, learning_rate=0.0075, print_cost=True)

def vector_to_digit(initial_predictions):
    """
    converts vectors of length 10 to a single digit where the position is 1
    :param initial_predictions: matrix of predictions
    :return: vector of predictions
    """

    # shape of parameter predictions is (10, 42000)

    pred_updated = np.zeros((1, m))

    for i in range(m):
        temp_pred = initial_predictions[:, i]
        pred_updated[:, i] = np.where(temp_pred == np.amax(temp_pred))[0]

    return pred_updated


if __name__ == "__main__":

    user = input("\nEnter command (train or test or acc (for accuracy)): ")

    while user != 'b':
        if user == 'train':
            parameters, costs = model.train()
            pickle_out = open("dict.pickle", "wb")
            pickle_cost = open("costs_place.picke", "wb")
            pickle.dump(costs, pickle_cost)
            pickle.dump(parameters, pickle_out)
            pickle_out.close()

        elif user == 'test':
            choice = input("\nNew digit or costs graph (n / c): ")

            if choice == 'n':
                pickle_in = open("dict.pickle", "rb")
                parameters = pickle.load(pickle_in)
                X_test = test_data[:, np.random.randint(28000)].reshape(784, 1)  # random image
                output = model.test(parameters, X_test)
                print(output)
                digit = np.where(output == np.amax(output))[0][0]
                fig = np.asarray(X_test).reshape(28, 28)
                plt.title(f"The test example digit is: {digit}")
                plt.imshow(fig)
                plt.show()
                plt.close()

            elif choice == 'c':
                pickle_in = open("costs_place.pickle", "rb")
                costs = pickle.load(pickle_in)
                fig = plt.plot(costs)
                plt.show()
                plt.close()

        elif user == 'acc': # for calculating accuracy over the training set
            pickle_in = open("dict.pickle", "rb")
            parameters = pickle.load(pickle_in)

            temp = model.test(parameters, train_data) # shape (10, 42000)
            predictions = vector_to_digit(temp) # shape (1, 42000)

            accuracy = test_accuracy(predictions)

            print(f"Accuracy on training set is: {accuracy}%")

        user = input("\nEnter command (train or test): ")

