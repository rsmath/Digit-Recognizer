# DIGIT RECOGNIZER [![HitCount](http://hits.dwyl.io/ramanshsharma2806/Digit-Recognizer.svg)](http://hits.dwyl.io/ramanshsharma2806/Digit-Recognizer)


## PROJECT DESCRIPTION

### Updates

As of December 8th, 2019, I have finished the purpose of this project. I wanted to implement a Vanilla Neural Network with basic Python, Numpy, and Pandas. I have even added mini-batch gradient descent and Adam optimization algorithm.

I am __*not*__ going to be adding regularization or other kinds of tuning to this neural network.

### Introduction


I recently finished the [Coursera Neural Networks and Deep Learning][1] course from [deeplearning.ai][2]. I am really excited to do this project and apply my knowledge of Vanilla Neural Networks now.

-------

### Methodology


I will be making my Vanilla NN in Python3 using the following libraries      **__(Note, this list is as of November 13th 2019, it may change as the project develops further)__**

* Numpy - To do all the major computations in the program
* Pandas - to load and store the image data
* Matplotlib - to graph different values over time to tune the hyperparameters


-------

### Network Architecture

The neural network will be a L-layer network. This means that I will be testing the best efficiency using different number of layers (having different number of neurons).

The architecture (layer dimensions) can be changed by the user in the [main.py](../master/src/main.py) module in the layers list.

The parameters that are initialized in the model are -:

- parameters = None 
- layer_dims = None 
- iterations = 3000
- learning_rate = 0.075
- mini_batch_size = 1024
- print_cost = False

All these variables can be passed in by the user, or left empty if the user wishes to use the values set in this project.

**I have implemented mini batch gradient descent and Adam Optimization algorithm in this project. Therefore, in just under 20 iterations, the model learns wonderfully.**

(*Note it is highly recommended that the iterations value is passed to be a number under 50, as it takes really long for the model to train given that libraries such as pytorch and tensorflow are not used. The set value of 3000 was initially set as it was in the deep learning course.*)



**Note: Please download the libraries using pip in order to run the required libraries for this project.**


```bash
pip install -r requirements.txt
```

**Note: Or download the libraries using conda like below if you use Anaconda**

```bash
conda env create -f environment.yml
```


Personally, conda environments run much better in PyCharm since they come preinstalled with most libraries. Go [here](https://www.anaconda.com/distribution/#download-section) to download Anaconda for yourself.



To initialize a model, download this repository, open an empty python module (in the same directory as the repository), and import as follows:


```python
from src.model import VanillaNN
import numpy as np

layers = [784, 30, 30, 10] # Do not change the first and the last values. 784 is the length of each image's pixel features and the output layer has 10 
# probability values

model = VanillaNN(layer_dims=layers, iterations=700, learning_rate=0.0025, print_cost=True)
# you can set these variables as per your own liking

# to train, just run the following line
parameters, train_costs = model.train()

# parameters can now be used to test any new image

# to test the digit of a new image
output = model.test(parameters, X_test) # where X_test is your image of length 784 pixel values
print(f"\nOutput probabilities are: \t\n{output}\n")
digit = np.where(output == np.amax(output))[0][0]
```

I have recently trained the model on my machine and uploaded the parameters on this repository using the **pickle** library.

If you would like to run the model without training it, run the [main.py](https://github.com/ramanshsharma2806/Digit-Recognizer/blob/master/src/main.py) file and pass in the *test* command as many times as you like!
(**Note**: when you pass in **test** or **c** (cost), in order to pass in your next command, close the window of the cost function image/predicted digit image)


-------

### Dataset

I will be using data from a MNIST handwritten digit recognition competition from [Kaggle][3].

Link - [https://www.kaggle.com/c/digit-recognizer/data](https://www.kaggle.com/c/digit-recognizer/data)

-------

### Exploring the training data

![Digits from 0-9](digits.png?raw=true "Digits from 0-9")

-------

### Results

I had to split the MNIST train set into 32000 (train) and 10000 (test) values since the actual test data did not have any ground truth labels.

Since I recently (*on December 9th*) applied [Adam optimization algorithm][4] and mini batch gradient descent to my model, I have achieved better results.

I achieved **98%** accuracy on the training set and **96%** accuracy on the test set. I do not see the model overfitting as of now, so I will have to further assess where exactly I might be able to improve the model in terms of accuracy.


-------

### Contact

If you have any concern or suggestion regarding this project, feel free to email me at [sharmar@bxscience.edu](sharmar@bxscience.edu).



[1]: https://www.coursera.org/learn/neural-networks-deep-learning/
[2]: https://www.coursera.org/specializations/deep-learning?
[3]: https://www.kaggle.com/
[4]: https://arxiv.org/pdf/1412.6980.pdf
