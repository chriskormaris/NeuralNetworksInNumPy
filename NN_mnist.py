# 1st Activation Function: tanh
# 2nd Activation Function: softmax
# Maximum Likelihood Estimate Function: Cross Entropy Function
# train algorithm: gradient ascent
# Bias terms are used.

# force the result of divisions to be float numbers
from __future__ import division

import numpy as np
import re
from pandas import DataFrame
import pandas as pd

__author__ = 'c.kormaris'

# ignore errors
np.seterr(all='ignore')

# set options
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)

###############


class NNParams:
    num_input_layers = 784  # D: number of nodes in the input layers (aka: no of features)
    num_hidden_layers = 100  # M: number of nodes in the hidden layer
    num_output_layers = 10  # K: number of nodes in the output layer (aka: no of categories)
    # Gradient ascent parameters
    eta = 0.01  # the learning rate for gradient ascent
    reg_lambda = 0.01  # the regularization parameter


###############

# FUNCTIONS #


def read_data(path, testOrTrainFile):
    text_file = open(path + testOrTrainFile + ".txt", "r")
    lines = text_file.readlines()
    text_file.close()

    X = [[0 for _ in range(NNParams.num_input_layers)] for _ in
         range(len(lines))]  # X: len(lines) x num_input_layers
    for i in range(len(lines)):
        tokens = lines[i].split(" ")
        for j in range(NNParams.num_input_layers):
            if j == NNParams.num_input_layers-1:
                tokens[j] = tokens[j].replace("\n", "")
            X[i][j] = np.float(tokens[j])

    X = np.array(X).astype(np.float)  # convert classification parameter to the appropriate data type
    return X


def read_labels(path, testOrTrainFile):
    text_file = open(path + testOrTrainFile + ".txt", "r")
    lines = text_file.readlines()
    text_file.close()

    digit = np.int(re.sub('[^0-9]', '', testOrTrainFile))
    y = [digit] * len(lines)
    y = np.array(y).astype(np.int)  # convert classification parameter to the appropriate data type
    return y

	
def tanh_output_to_derivative(output):
    return 1 - np.square(output)


def softmax(x):
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=1, keepdims=True))


# concat ones column vector as the first column of the matrix
def concat_ones_vector(x):
    ones_vector = np.ones((x.shape[0], 1))
    return np.concatenate((ones_vector, x), axis=1)


# Helper function to evaluate the likelihood on the train dataset.
def likelihood(W1, W2, X, t):
    num_examples = len(X)  # N: training set size

    # Forward propagation to calculate our predictions
    s1 = X.dot(W1.T)
    o1 = np.tanh(s1)
    o1 = concat_ones_vector(o1)
    s2 = o1.dot(W2.T)

    # Calculating the mle
    mle = sum(sum(np.multiply(t, s2)))  # NxK .* NxK

    # Add regularization term to likelihood (optional)
    mle -= NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return mle


def test(W1, W2, X):
    # Forward propagation
    s1 = X.dot(W1.T)
    o1 = np.tanh(s1)
    o1 = concat_ones_vector(o1)
    s2 = o1.dot(W2.T)
    o2 = softmax(s2)
    return np.argmax(o2, axis=1)


# This function learns the parameter weights W1, W2 for the neural network and returns them.
# - iterations: Number of iterations through the training data for gradient ascent
# - print_estimate: If True, print the estimate every 1000 iterations
def train(X, y, iterations=20000, print_estimate=False):
    t = np.zeros((y.shape[0], NNParams.num_output_layers))
    t[np.arange(y.shape[0]), y] = 1  # t: 1-hot matrix for the categories y
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(NNParams.num_hidden_layers, NNParams.num_input_layers) / np.sqrt(NNParams.num_input_layers)  # W1: MxD
    W2 = np.random.randn(NNParams.num_output_layers, NNParams.num_hidden_layers) / np.sqrt(NNParams.num_hidden_layers)  # W2: KxM

    # concat ones vector
    W1 = concat_ones_vector(W1)  # W1: MxD+1
    W2 = concat_ones_vector(W2)  # W2: KxM+1

    # Run Batch gradient ascent
    for i in range(iterations):

        # W1: MxD+1 = num_hidden_layers x num_of_features
        # W2: KxM+1 = num_of_categories x num_hidden_layers

        # Forward propagation
        s1 = X.dot(W1.T)  # s1: NxM
        o1 = np.tanh(s1)  # o1: NxM
        grad = tanh_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM
        o1 = concat_ones_vector(o1)  # o1: NxM+1
        s2 = o1.dot(W2.T)  # s2: NxK
        o2 = softmax(s2)  # o2: NxK

        # Backpropagation
        delta1 = t - o2  # delta1: NxK
        W2_reduce = W2[np.ix_(np.arange(W2.shape[0]), np.arange(1, W2.shape[1]))]  # skip the first column of W2: KxM
        delta2 = np.dot(delta1, W2_reduce)  # delta2: NxM
        delta3 = np.multiply(delta2, grad)  # element-wise multiplication, delta3: NxM

        dW1 = np.dot(delta3.T, X)  # MxD+1
        dW2 = np.dot(delta1.T, o1)  # KxM+1

        # Add regularization terms
        dW1 += -NNParams.reg_lambda * W1
        dW2 += -NNParams.reg_lambda * W2

        # Update gradient ascent parameters
        W1 += NNParams.eta * dW1
        W2 += NNParams.eta * dW2

        # Optionally print the estimate.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        #if print_estimate and i % 1000 == 0:
        if print_estimate:
            print("Likelihood estimate after iteration %i: %f" % (i, likelihood(W1, W2, X, t)))

    return W1, W2


###############

# MAIN #

mnist_dir = "./mnisttxt/"

print("Reading TRAIN files...")
X_train = np.array  # 2d array
y_train = np.array  # 1d array
for i in range(10):
    print("Reading " + "'train" + str(i) + ".txt'")
    if i == 0:  # read the first file
        X_train = read_data(mnist_dir, 'train' + str(i))
        y_train = read_labels(mnist_dir, 'train' + str(i))
    else:
        X_train = np.concatenate((X_train, read_data(mnist_dir, 'train' + str(i))), axis=0)
        y_train = np.concatenate((y_train, read_labels(mnist_dir, 'train' + str(i))), axis=0)

'''
print("\n Xtrain:")
df = DataFrame(X_train)
df.index = range(X_train.shape[0])
df.columns = range(X_train.shape[1])
print(df)

print("ytrain: " + str(y_train))
'''

print("\n")

print("Reading TEST files...")
X_test = np.array  # 2d array
y_test_true = np.array  # 1d array
for i in range(10):
    print("Reading " + "'test" + str(i) + ".txt'")
    if i == 0:  # read the first file
        X_test = read_data(mnist_dir, 'test' + str(i))
        y_test_true = read_labels(mnist_dir, 'test' + str(i))
    else:
        X_test = np.concatenate((X_test, read_data(mnist_dir, 'test' + str(i))), axis=0)
        y_test_true = np.concatenate((y_test_true, read_labels(mnist_dir, 'test' + str(i))), axis=0)

'''
print("\n Xtest:")
df = DataFrame(X_test)
df.index = range(X_test.shape[0])
df.columns = range(X_test.shape[1])
print(df)

print("y_test_true: " + str(y_test_true))
'''

print("\n")

# concat ones vector
X_train = concat_ones_vector(X_train)
X_test = concat_ones_vector(X_test)

# normalize the data using range normalization
X_train = X_train / 255
X_test = X_test / 255

# define the learning rate based on the number of train data
NNParams.eta = 0.5 / len(X_train)

# train the Neural Network Model
W1, W2 = train(X_train, y_train, iterations=500, print_estimate=True)

# test the Neural Network Model
predicted = test(W1, W2, X_test)

# check predictions
wrong_counter = 0  # the number of wrong classifications made by the Neural Network

print("\n")
print('checking predictions...')
for i in range(len(predicted)):
    if predicted[i] == y_test_true[i]:
        print("data " + str(i) + ' classified as: ' + str(predicted[i]) + ' -> correct')
    elif predicted[i] != y_test_true[i]:
        print("data " + str(i) + ' classified as: ' + str(predicted[i]) + ' -> WRONG!')
        wrong_counter = wrong_counter + 1

print('\n')

# Accuracy

accuracy = ((len(X_test) - wrong_counter) / len(X_test)) * 100
print("accuracy: " + str(accuracy) + " %")
print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(X_test)) + ' images!')
