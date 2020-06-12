"""
This module is just to run the model in a clean environment with new data
"""

import pickle
from src.model import VanillaNN, test_accuracy
from src.prep_data import test_data, train_data, m_train, m_test, labels_train, labels_test
from matplotlib import pyplot as plt
import numpy as np
from src.new_image import image
import time


parameters, train_costs = None, None

layers = [784, 30, 30, 10]

epochs = 15

batch_size = 1024

model = VanillaNN(layer_dims=layers, iterations=epochs, learning_rate=0.0025, mini_batch_size=batch_size,
                  print_cost=True)


def vector_to_digit(initial_predictions, size=None):
    """
    converts vectors of length 10 to a single digit where the position is 1
    :param size: number of examples in data set, either for training or test set
    :param initial_predictions: matrix of predictions
    :return: vector of predictions
    """

    # shape of parameter predictions is (10, 32000)

    if size is None:
        size = m_train  # default is for training set

    pred_updated = np.zeros((1, size))

    for i in range(size):
        temp_pred = initial_predictions[:, i]
        pred_updated[:, i] = np.where(temp_pred == np.amax(temp_pred))[0]

    return pred_updated


command_message = "\nList of commands are:" \
                  "\ne:          terminate the program" \
                  "\nnew :       test the new .png image you have inserted" \
                  "\ntrain adam: train the model on adam optimization algorithm" \
                  "\ntrain gd:   train the model on gradient descent algorithm" \
                  "\nc:          display the cost function of training and validation sets" \
                  "\nacc:        print the accuracies for training and test sets" \
                  "\ntest:       test a new random image from the test set and classify it" \
                  "\ncommands:   print this command list\n"

print(command_message)

if __name__ == "__main__":

    user = input("Enter command: ")

    while user != 'e':
        if user == 'train adam':
            start = time.time()
            parameters, train_costs, cv_costs = model.train(X=train_data, technique='adam')
            end = time.time()
            print(f"\nTime taken for {epochs} epochs: {end - start} seconds \n")
            pickle_out = open("dict.pickle", "wb")
            pickle_cost = open("costs_place.pickle", "wb")
            pickle_cvcosts = open("cv_costs.pickle", "wb")
            pickle.dump(cv_costs, pickle_cvcosts)
            pickle.dump(train_costs, pickle_cost)
            pickle.dump(parameters, pickle_out)
            pickle_out.close()
            pickle_cost.close()
            pickle_cvcosts.close()

        elif user == 'train gd':
            start = time.time()
            parameters, train_costs, cv_costs = model.train(X=train_data, technique='gd')
            end = time.time()
            print(f"\nTime taken for 50 epochs: {end - start} seconds \n")
            pickle_out = open("dict.pickle", "wb")
            pickle_cost = open("costs_place.pickle", "wb")
            pickle_cvcosts = open("cv_costs.pickle", "wb")
            pickle.dump(cv_costs, pickle_cvcosts)
            pickle.dump(train_costs, pickle_cost)
            pickle.dump(parameters, pickle_out)
            pickle_out.close()
            pickle_cost.close()
            pickle_cvcosts.close()

        elif user == "new":
            pickle_in = open("dict.pickle", "rb")
            parameters = pickle.load(pickle_in)
            X_test = image
            output = model.test(parameters, X_test)
            print(f"\nOutput probabilities are: \t\n{output}\n")
            digit = np.where(output == np.amax(output))[0][0]
            fig = np.asarray(X_test).reshape(28, 28)
            plt.title(f"The test example digit is: {digit}")
            plt.imshow(fig)
            plt.show()
            plt.close()

        elif user == 'test':
            pickle_in = open("dict.pickle", "rb")
            parameters = pickle.load(pickle_in)
            X_test = test_data[:, np.random.randint(m_test)].reshape(784, 1)  # random image
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
            pickle_cv = open("cv_costs.pickle", "rb")

            train_costs = pickle.load(pickle_inc)
            cv_costs = pickle.load(pickle_cv)

            width_in_inches = 30
            height_in_inches = 15
            dots_per_inch = 50

            plt.figure(
                figsize=(width_in_inches, height_in_inches),
                dpi=dots_per_inch)

            plt.plot(train_costs, '^:r', label="train", mew=7, linewidth=3)
            plt.plot(cv_costs, '^:b', label="validation", mew=7, linewidth=3)
            plt.legend(loc="upper right", fontsize=25)

            plt.title("Cost (train and validation) as the model trains", fontsize=35, color='black')

            plt.xlabel('Epoch', fontsize=35, color='black')
            plt.ylabel("Cost", fontsize=35, color='black')

            plt.xticks(range(0, len(train_costs) + 1), fontsize=17, color='black')
            plt.yticks(fontsize=17, color='black')

            plt.show()
            plt.close()

        elif user == 'acc':  # for calculating train_accuracy over the training set
            pickle_in = open("dict.pickle", "rb")
            parameters = pickle.load(pickle_in)

            temp_train = model.test(parameters, train_data)  # shape (10, 32000)
            train_predictions = vector_to_digit(temp_train, size=m_train)  # shape (1, 32000)

            temp_test = model.test(parameters, test_data)  # shape (10, 10000)
            test_predictions = vector_to_digit(temp_test, size=m_test)  # shape (1, 10000)

            train_accuracy = test_accuracy(train_predictions, ground_truth=labels_train, size=m_train)
            tests_accuracy = test_accuracy(test_predictions, ground_truth=labels_test, size=m_test)

            print(f"\nAccuracy on training set is: {train_accuracy}%")
            print(f"Accuracy on test set is: {tests_accuracy}%\n")

        elif user == 'commands':
            print(command_message)

        user = input("Enter command: ")

    print("\nSee you later user.")
