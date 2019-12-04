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
train_data = data_original[:, : 29400]  # (784, 29400) since test.csv does not have any labels, 29400 of train is the
# training set, remaining is test set

labels_train = labels[: 29400] # labels for training set


test_data = data_original[:, 29400:]  # shape (784, 12600)

labels_test = labels[29400:] # labels for test set

m_train = len(labels_train)  # number of training examples in training data
m_test = len(labels_test) # number of examples in test set

y = []
for i in range(m_train):
    temp = np.zeros((10,), dtype=int)
    temp[labels_train[i]] = 1
    y.append(temp)

y = np.asarray(y)  # shape (29400, 10), a 1 for each label digit's position in an empty (10,) zeros array
y = np.transpose(y)  # fixes a bug and changes shape of y to (10, 29400)

'''
FOR NOW THERE WILL BE NO CROSS VALIDATION TEST. AFTER A PRELIMINARY MODEL HAS BEEN MADE AND TESTED, 
THEN A CROSS VALIDATION TEST WILL BE USED TO OPTIMIZE THE REGULARIZATION PARAMETER (THERE IS NO
REGULARIZATION RIGHT NOW EITHER)
'''
