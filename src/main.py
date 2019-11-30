"""
This module is just to run the model in a clean environment with new data
"""

from src.model import VanillaNN
from src.prep_data import test_data, train_data
from matplotlib import pyplot as plt
import numpy as np


layers = [784, 35, 25, 20, 10]

model = VanillaNN(layer_dims=layers, iterations=100, learning_rate=0.1, print_cost=True)

if __name__ == "__main__":
    costs = model.train()
    X_test = test_data[:, np.random.randint(28000)].reshape(784, 1) # random image
    output = model.test(X_test)
    print(output)
    digit = np.where(output == np.amax(output))[0][0]
    fig = np.asarray(X_test).reshape(28, 28)
    plt.title(f"The test example digit is: {digit}")
    # fig = plt.plot(costs)
    plt.imshow(fig)
    plt.show()






