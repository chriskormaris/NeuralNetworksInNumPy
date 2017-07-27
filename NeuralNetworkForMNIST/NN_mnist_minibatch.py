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
    batch_size = 500


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


# activation function #1
def h1(x):
    return np.log(1 + np.exp(x))


# activation function #1 derivative / the same as the sigmoid function
def h1_output_to_derivative(output):
    return sigmoid(output)


# activation function #2 derivative
def tanh_output_to_derivative(output):
    return 1 - np.square(output)


# activation function #3 derivative
def cos_output_to_derivative(output):
    return -np.sin(output)


def sigmoid(x):
    return np.matrix(1 / (1 + np.exp(-x)))


# activation function for the 2nd layer
def softmax(x):
    #return np.divide(np.exp(x), np.sum(np.exp(x), axis=1, keepdims=True))
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=1))


# concat ones column vector as the first column of the matrix (adds bias term)
def concat_ones_vector(x):
    ones_vector = np.ones((x.shape[0], 1))
    return np.concatenate((ones_vector, x), axis=1)


# Forward propagation
def forward(X, W1, W2):
    s1 = X.dot(W1.T)  # s1: NxM

    # activation function #1
    #o1 = np.tanh(s1)  # o1: NxM
    #grad = tanh_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM

    # activation function #2
    o1 = np.tanh(s1)  # o1: NxM
    grad = tanh_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM

    # activation function #3
    #o1 = np.cos(s1)  # o1: NxM
    #grad = cos_output_to_derivative(o1)  # the gradient of cos function, grad: NxM

    o1 = concat_ones_vector(o1)  # o1: NxM+1
    s2 = o1.dot(W2.T)  # s2: NxK
    o2 = softmax(s2)  # o2: NxK
    return s1, o1, grad, s2, o2


# Helper function to evaluate the likelihood on the train dataset.
def likelihood(X, t, W1, W2):
    num_examples = len(X)  # N: training set size

    # Forward propagation to calculate our predictions
    _, _, _, s2, _ = forward(X, W1, W2)

    # Calculating the mle
    mle = np.sum(np.sum(np.multiply(t, s2)))  # NxK .* NxK

    # Add regularization term to likelihood (optional)
    mle -= NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return mle


def test(W1, W2, X):
    # Forward propagation
    _, _, _, _, o2 = forward(X, W1, W2)
    return np.argmax(o2, axis=1)


# Train using Stochastic Gradient Ascent
def train(X, y, epochs=50, tol=1e-6, print_estimate=False):
    t = np.zeros((y.shape[0], NNParams.num_output_layers))
    t[np.arange(y.shape[0]), y] = 1  # t: 1-hot matrix for the categories y
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(NNParams.num_hidden_layers, NNParams.num_input_layers) / np.sqrt(
        NNParams.num_input_layers)  # W1: MxD
    W2 = np.random.randn(NNParams.num_output_layers, NNParams.num_hidden_layers) / np.sqrt(
        NNParams.num_hidden_layers)  # W2: KxM

    # concat ones vector
    W1 = concat_ones_vector(W1)  # W1: MxD+1
    W2 = concat_ones_vector(W2)  # W2: KxM+1

    # Run Stochastic gradient ascent
    num_examples = X.shape[0]
    s_old = -np.inf
    for e in range(epochs):

        s = 0
        iterations = int(np.ceil(num_examples / NNParams.batch_size))
        for i in range(iterations):
            start_index = int(i * NNParams.batch_size)
            end_index = int(i * NNParams.batch_size + NNParams.batch_size - 1)
            W1, W2 = grad_ascent(np.matrix(X[start_index:end_index, :]), np.matrix(t[start_index:end_index, :]), W1, W2)
            s = s + likelihood(np.matrix(X[start_index:end_index, :]), np.matrix(t[start_index:end_index, :]), W1, W2)

        # Optionally print the estimate.
        if print_estimate:
            print("Likelihood estimate after epoch %i: %f" % (e, s))

        if np.abs(s - s_old) <= tol:
            break

        s_old = s


    return W1, W2


# Update the Weight matrices using Gradient Ascent
def grad_ascent(X, t, W1, W2):
    # W1: MxD+1 = num_hidden_layers x num_of_features
    # W2: KxM+1 = num_of_categories x num_hidden_layers

    # Forward propagation
    _, o1, grad, s2, o2 = forward(X, W1, W2)

    # Backpropagation
    delta1 = t - o2  # delta1: 1xK
    W2_reduce = W2[np.ix_(np.arange(W2.shape[0]), np.arange(1, W2.shape[1]))]  # skip the first column of W2: KxM
    delta2 = np.dot(delta1, W2_reduce)  # delta2: 1xM
    delta3 = np.multiply(delta2, grad)  # element-wise multiplication, delta3: 1xM

    dW1 = np.dot(delta3.T, X)  # MxD+1
    dW2 = np.dot(delta1.T, o1)  # KxM+1

    # Add regularization terms
    dW1 = dW1 - NNParams.reg_lambda * W1
    dW2 = dW2 - NNParams.reg_lambda * W2

    # Update gradient ascent parameters
    W1 = W1 + NNParams.eta * dW1
    W2 = W2 + NNParams.eta * dW2

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
W1, W2 = train(X_train, y_train, epochs=50, tol=1e-6, print_estimate=True)

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
