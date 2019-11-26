"""
This module is just to run the model in a clean environment with new data
"""

from src.model import VanillaNN
from src.prep_data import test_data
from matplotlib import pyplot as plt
import numpy as np

model = VanillaNN(iterations=1000, learning_rate=0.01, print_cost=True)

if __name__ == "__main__":
    model.train()
    X_test = test_data[np.random.randint(28000)].reshape(784, 1) # random image
    output = model.test(X_test)
    digit = np.where(output == np.amax(output))[0][0]
    fig = np.asarray(X_test).reshape(28, 28)
    plt.title(f"The test example digit is: {digit}")
    plt.imshow(fig)
    plt.show()






