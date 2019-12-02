"""
This module is just to run the model in a clean environment with new data
"""


import pickle
from src.model import VanillaNN, test_accuracy
from src.prep_data import test_data, train_data, m_train, m_test, labels_train, labels_test
from matplotlib import pyplot as plt
import numpy as np


parameters, train_costs = None, None
layers = [784, 30, 30, 10]

model = VanillaNN(layer_dims=layers, iterations=700, learning_rate=0.0025, print_cost=True)

def vector_to_digit(initial_predictions, size=None):
    """
    converts vectors of length 10 to a single digit where the position is 1
    :param size: number of examples in dataset, either for training or test set
    :param initial_predictions: matrix of predictions
    :return: vector of predictions
    """

    # shape of parameter predictions is (10, 32000)

    if size is None:
        size = m_train # default is for training set

    pred_updated = np.zeros((1, size))

    for i in range(size):
        temp_pred = initial_predictions[:, i]
        pred_updated[:, i] = np.where(temp_pred == np.amax(temp_pred))[0]

    return pred_updated


if __name__ == "__main__":

    user = input("\nEnter command (b (to terminate) or train or test or c (for train_costs) or acc (for "
                 "train_accuracy)): ")

    while user != 'b':
        if user == 'train':
            parameters, train_costs = model.train(X=train_data)
            pickle_out = open("dict.pickle", "wb")
            pickle_cost = open("costs_place.pickle", "wb")
            pickle.dump(train_costs, pickle_cost)
            pickle.dump(parameters, pickle_out)
            pickle_out.close()
            pickle_cost.close()

        elif user == 'test':
            pickle_in = open("dict.pickle", "rb")
            parameters = pickle.load(pickle_in)
            X_test = test_data[:, np.random.randint(10000)].reshape(784, 1)  # random image
            output = model.test(parameters, X_test)
            print(f"\nOutput probabilities are: \t\n{output}\n")
            digit = np.where(output == np.amax(output))[0][0]
            fig = np.asarray(X_test).reshape(28, 28)
            plt.title(f"The test example digit is: {digit}")
            plt.imshow(fig)
            plt.show()
            plt.close()

        elif user == 'c':
            pickle_inc = open("costs_place.pickle", "rb")
            train_costs = pickle.load(pickle_inc)
            fig1 = plt.plot(train_costs)
            plt.title("Cost as the model trains")
            plt.xlabel('Iterations')
            plt.ylabel("Cost")
            plt.show()
            plt.close()

        elif user == 'acc': # for calculating train_accuracy over the training set
            pickle_in = open("dict.pickle", "rb")
            parameters = pickle.load(pickle_in)

            temp_train = model.test(parameters, train_data) # shape (10, 32000)
            train_predictions = vector_to_digit(temp_train, size=m_train) # shape (1, 32000)

            temp_test = model.test(parameters, test_data) # shape (10, 10000)
            test_predictions = vector_to_digit(temp_test, size=m_test) # shape (1, 10000)

            train_accuracy = test_accuracy(train_predictions, ground_truth=labels_train, size=m_train)
            test_accuracy = test_accuracy(test_predictions, ground_truth=labels_test, size=m_test)

            print(f"\nAccuracy on training set is: {train_accuracy}%")
            print(f"Accuracy on test set is: {test_accuracy}%")

        user = input("\nEnter command (b (to terminate) or train or test or c (for train_costs) or acc (for "
                     "train_accuracy)): ")


