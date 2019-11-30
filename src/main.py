"""
This module is just to run the model in a clean environment with new data
"""

from src.model import VanillaNN
from src.prep_data import test_data, train_data
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


layers = [784, 40, 40, 10]

model = VanillaNN(layer_dims=layers, iterations=800, learning_rate=0.01, print_cost=True)

def train():
    parameters, costs = model.train()
    pd.DataFrame.from_dict(data=parameters, orient='index').to_csv('dict_file.csv', header=False)

def test():
    X_test = test_data[:, np.random.randint(28000)].reshape(784, 1)  # random image
    parameters = pd.read_csv('dict_file.csv', header=None, index_col=0, squeeze=True).to_dict()
    output = model.test(parameters, X_test)
    print(output)
    digit = np.where(output == np.amax(output))[0][0]
    fig = np.asarray(X_test).reshape(28, 28)
    plt.title(f"The test example digit is: {digit}")
    # fig = plt.plot(costs)
    plt.imshow(fig)
    plt.show()


if __name__ == "__main__":
    user = input("\nEnter \'train\' for training and \'test\' for testing: ")
    
    while (user != 'train' and user != 'test') or user != 'b':
        user = input("\nEnter \'train\' for training and \'test\' for testing: ")
    
    if user == 'train':
        train()
    
    elif user == test():
        test()

