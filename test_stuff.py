

# this file is just for testing out stuff

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# training_data = pd.read_csv("../digit-recognizer/train.csv", index_col=0)
# test_data = pd.read_csv("../digit-recognizer/test.csv")
# sample_submission = pd.read_csv("../digit-recognizer/sample_submission.csv", index_col='ImageId')


# print(training_data.shape) # 42000x784
#
#
# print(test_data.shape) # 28000, 784
#
# print(sample_submission.shape) # 28000, 2
#
# print(sample_submission.head())
#
#
# print('\b\b\b\b\b\b\b\b\b\b\b')
#
# print(str(0.6 * 70) + " " + str(0.2 * 70) + " " + str(0.2 * 70))
#
#
# temp = np.zeros((10,), dtype=int)
# print(temp)
#

# vars = []
# L = 10
#
# for i in range(1, 10):
#     print(i)
#     vars.append(i)
#
# print(vars)


grads = [(1, 2), (4, 5), (5, 6)]


# print(grads[0][1])


# for i in range(20):
#     d = np.random.randint(5)
#     print(d)

#
# f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 6, 11])
#
# d = np.where(f == np.amax(f))
# print(d[0])


a = np.arange(1, 11)
print(f"a: {a}")
b = np.array(a, copy=True)

print(f"b: {b}")

a = np.arange(5, 122)

print(f"a now: {a}")
print(f"b now: {b}")

