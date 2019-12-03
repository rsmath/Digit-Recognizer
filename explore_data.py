"""
Python file for exploring the data
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


training_data = pd.read_csv("../digit-recognizer/train.csv", index_col=0) # loading the training training_data

# top of the training_data
print(training_data.head())


print('42000 examples, each with 784 pixel values')

print(f"shape of training data: {training_data.shape}")

plt.rcParams['figure.figsize'] = (15.0, 15.0)  # set default size of plots

for i in range(0, 10):
    # for plotting multiple plots in same cell
    plt.subplot(3, 4, i + 1)

    plt.title(f"Example of digit: {i}")

    # acquiring a random digit's random example pixel values
    obj = np.transpose(training_data.loc[i].iloc[np.random.randn(1)])

    # reshaping the training_data type to a square image
    obj = np.asarray(obj).reshape(28, 28)

    plt.imshow(obj)  # plotting using matplotlib

plt.savefig('digits.png')


