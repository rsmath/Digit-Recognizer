'''
This file will be used to prepare the data to be used by the model.
Data will be divided into train, cross validation, and test
'''

import numpy as np
import pandas as pd

train_data = np.asarray(pd.read_csv('../digit-recognizer/train.csv', index_col=0))
test_data = np.asarray(pd.read_csv('../digit-recognizer/test.csv'))


'''
FOR NOW THERE WILL BE NO CROSS VALIDATION TEST. AFTER A PRELIMINARY MODEL HAS BEEN MADE AND TESTED, 
THEN A CROSS VALIDATION TEST WILL BE USED TO OPTIMIZE THE REGULARIZATION PARAMETER (THERE IS NO
REGULARIZATION RIGHT NOW EITHER)
'''


