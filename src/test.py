

# this file is just for testing out stuff

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

training_data = pd.read_csv("../digit-recognizer/train.csv", index_col=0)
test_data = pd.read_csv("../digit-recognizer/test.csv")
sample_submission = pd.read_csv("../digit-recognizer/sample_submission.csv", index_col='ImageId')


print(training_data.shape) # 42000x784


print(test_data.shape) # 28000, 784

print(sample_submission.shape) # 28000, 2

print(sample_submission.head())
