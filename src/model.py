'''
The main model file. Here all the different functions will be pieced together to form the multilayer Neural Network.
If the size of the NN is desired to be changed, it can be done in the layer_dims array.
'''

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (15.0, 15.0) # set default size of plots


layers = [784, 25, 7, 5, 10] # four layer model, 4th layer having 10 output units which will be rounded off and highest
# probability will be the predicted digit

layer_dims = np.array(layers)









