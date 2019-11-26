"""
This module is just to run the model in a clean environment with new data
"""

from src.model import VanillaNN
from src.prep_data import test_data
import numpy as np

model = VanillaNN(iterations=1000, learning_rate=0.01, print_cost=True)

if __name__ == "__main__":
    model.train()
    model.test(test_data[np.random.randint(28000)].reshape(784, 1)) # random image






