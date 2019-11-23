'''
This file will be used to prepare the data to be used by the model.
Data will be divided into train, cross validation, and test
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train_data = np.asarray(pd.read_csv('../digit-recognizer/train.csv', index_col=0))
test_data = np.asarray(pd.read_csv('../digit-recognizer/test.csv'))

# acquiring a random digit's random example pixel values
obj = train_data[0, :]

# reshaping the training_data type to a square image
obj = np.asarray(obj).reshape(28, 28)

plt.imshow(obj) # plotting using matplotlib
'''
FOR NOW THERE WILL BE NO CROSS VALIDATION TEST. AFTER A PRELIMINARY MODEL HAS BEEN MADE AND TESTED, 
THEN A CROSS VALIDATION TEST WILL BE USED TO OPTIMIZE THE REGULARIZATION PARAMETER (THERE IS NO
REGULARIZATION RIGHT NOW EITHER)
'''


