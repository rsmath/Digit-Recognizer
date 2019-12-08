"""
This file will be used to prepare the data to be used by the model.
Data will be divided into train, cross validation, and test
"""


import numpy as np
import pandas as pd


data = pd.read_csv('../digit-recognizer/train.csv')

labels = np.asarray(data['label'])  # can be used as y, the true values

data.set_index('label', inplace=True)  # permanently sets label column to be the index

data_original = np.asarray(data)
data_original = np.transpose(data_original)  # this fixes a small bug, changes shape to (784, 42000)

train_data = data_original[:, : 30000]  # (784, 30000) training data
labels_train = labels[: 30000] # labels for training set

cv_data = data_original[:, 30000:35000] # validation set will be used for plotting CV costs
labels_cv = labels[30000:35000] # labels for validation set

test_data = data_original[:, 35000:]  # shape (784, 7000)
labels_test = labels[35000:] # labels for test set

m_train = len(labels_train)  # number of training examples in training data
m_cv = len(labels_cv) # number of examples in validation set
m_test = len(labels_test) # number of examples in test set

y_cv = []
for i in range(m_cv):
    temp = np.zeros((10,), dtype=int)
    temp[labels_cv[i]] = 1
    y_cv.append(temp)

y_cv = np.asarray(y_cv)
y_cv = np.transpose(y_cv) # ground truth for validation test shape (10, 5000)

y = []
for i in range(m_train):
    temp = np.zeros((10,), dtype=int)
    temp[labels_train[i]] = 1
    y.append(temp)

y = np.asarray(y)  # shape (29400, 10), a 1 for each label digit's position in an empty (10,) zeros array
y = np.transpose(y)  # fixes a bug and changes shape of y to (10, 30000)


