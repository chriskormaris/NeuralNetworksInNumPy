# 1st Activation Function: tanh
# 2nd Activation Function: softmax
# Maximum Likelihood Estimate Function: Cross Entropy Function
# Train Algorithm: Batch Gradient Ascent
# Bias terms are used.

# force the result of divisions to be float numbers
from __future__ import division

import numpy as np

__author__ = 'c.kormaris'

# ignore errors
np.seterr(all='ignore')

# set options
#pd.set_option('display.width', 1000)
#pd.set_option('display.max_rows', 200)

###############


class NNParams:
    num_input_layers = 784  # D: number of nodes in the input layers (aka: no of features)
    num_hidden_layers = 100  # M: number of nodes in the hidden layer
    num_output_layers = 10  # K: number of nodes in the output layer (aka: no of categories)
    # Gradient ascent parameters
    eta = 0.1  # the learning rate for gradient ascent; it is changed inside the main
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
            X[i][j] = np.int(tokens[j])

    X = np.matrix(X)  # convert classification parameter to the appropriate data type
    return X


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
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=1))


# concat ones column vector as the first column of the matrix (adds bias term)
def concat_ones_vector(x):
    ones_vector = np.ones((x.shape[0], 1))
    return np.concatenate((ones_vector, x), axis=1)


# Feed-Forward
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
    #num_examples = len(X)  # N: training set size

    # Feed-Forward to calculate our predictions
    _, _, _, s2, _ = forward(X, W1, W2)

    A = s2
    K = NNParams.num_output_layers

    # Calculating the mle using the logsumexp trick
    maximum = np.max(A, axis=1)
    mle = np.sum(np.multiply(t, A)) - np.sum(maximum, axis=0) \
          - np.sum(np.log(np.sum(np.exp(A - np.repeat(maximum, K, axis=1)), axis=1)))
    #mle = np.sum(np.multiply(t, np.log(o2)))

    # Add regularization term to likelihood (optional)
    mle -= NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return mle


def test(X, W1, W2):
    # Feed-Forward
    _, _, _, _, o2 = forward(X, W1, W2)
    return np.argmax(o2, axis=1)


# Train using Batch Gradient Ascent
# This function learns the parameter weights W1, W2 for the neural network and returns them.
# - iterations: Number of iterations through the training data for gradient ascent.
# - print_estimate: If True, print the estimate every 1000 iterations.
def train(X, t, W1, W2, iterations=500, tol=1e-6, print_estimate=False):

    # Run Batch Gradient Ascent
    loss_old = -np.inf
    for i in range(iterations):

        W1, W2, _, _ = grad_ascent(X, t, W1, W2)

        # Optionally print the estimate.
        # This is expensive because it uses the whole dataset.
        if print_estimate:
            loss = likelihood(X, t, W1, W2)
            print("Likelihood estimate after iteration %i: %f" % (i, loss))
            if np.abs(loss - loss_old) < tol:
                break
            loss_old = loss

    return W1, W2


# Update the Weight matrices using Gradient Ascent
def grad_ascent(X, t, W1, W2):
    # W1: MxD+1 = num_hidden_layers x num_of_features
    # W2: KxM+1 = num_of_categories x num_hidden_layers

    # Feed-Forward
    _, o1, grad, s2, o2 = forward(X, W1, W2)

    # Back-Propagation
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

    return W1, W2, dW1, dW2


def gradient_check(X, t, W1, W2):
    _, _, gradEw1, gradEw2 = grad_ascent(X, t, W1, W2)
    epsilon = 1e-6

    # gradient_check for parameter W1
    numgradEw1 = np.zeros(W1.shape)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1tmp = W1
            W1tmp[i, j] = W1[i, j] + epsilon
            Ewplus = likelihood(X, t, W1tmp, W2)

            W1tmp = W1
            W1tmp[i, j] = W1[i, j] - epsilon
            Ewminus = likelihood(X, t, W1tmp, W2)

            numgradEw1[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff1 = np.sum(np.abs(gradEw1 - numgradEw1)) / np.sum(np.abs(gradEw1))
    print('The maximum absolute norm for parameter W1, in the gradient_check is: ' + str(diff1))

    # gradient_check for parameter W2
    numgradEw2 = np.zeros(W2.shape)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2tmp = W2
            W2tmp[i, j] = W2[i, j] + epsilon
            Ewplus = likelihood(X, t, W1, W2tmp)

            W2tmp = W2
            W2tmp[i, j] = W2[i, j] - epsilon
            Ewminus = likelihood(X, t, W1, W2tmp)

            numgradEw2[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff2 = np.sum(np.abs(gradEw2 - numgradEw2)) / np.sum(np.abs(gradEw2))
    print('The maximum absolute norm for parameter W2, in the gradient_check is: ' + str(diff2))


###############

# MAIN #

mnist_dir = "./mnisttxt/"

print("Reading TRAIN files...")
# read train images for digits 0,1, 2 and 3
X_train = np.matrix  # 2D matrix
y_train = np.matrix  # 1D matrix
for i in range(NNParams.num_output_layers):
    print("Reading " + "'train" + str(i) + ".txt'")
    X_train_class_i = read_data(mnist_dir, 'train' + str(i))
    N_train_i = X_train_class_i.shape[0]
    y_train_class_i = np.repeat([i], N_train_i, axis=0)
    if i == 0:
        X_train = X_train_class_i
        y_train = y_train_class_i
    else:
        X_train = np.concatenate((X_train, X_train_class_i), axis=0)
        y_train = np.concatenate((y_train, y_train_class_i), axis=0)

print('')

print("Reading TEST files...")
# read test images for digits 0,1, 2 and 3
X_test = np.matrix  # 2D matrix
y_test_true = np.matrix  # 1D matrix
for i in range(NNParams.num_output_layers):
    print("Reading " + "'test" + str(i) + ".txt'")
    X_test_class_i = read_data(mnist_dir, 'test' + str(i))
    N_test_i = X_test_class_i.shape[0]
    y_test_true_class_i = np.repeat([i], N_test_i, axis=0)
    if i == 0:
        X_test = X_test_class_i
        y_test_true = y_test_true_class_i
    else:
        X_test = np.concatenate((X_test, X_test_class_i), axis=0)
        y_test_true = np.concatenate((y_test_true, y_test_true_class_i), axis=0)

print('')

# normalize the data using range normalization
X_train = X_train / 255
X_test = X_test / 255

# concat ones vector
X_train = concat_ones_vector(X_train)
X_test = concat_ones_vector(X_test)

# construct t: 1-hot matrix for the categories y_train
t = np.zeros((y_train.shape[0], NNParams.num_output_layers))
t[np.arange(y_train.shape[0]), y_train] = 1

# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
W1 = np.random.randn(NNParams.num_hidden_layers, NNParams.num_input_layers) / np.sqrt(
    NNParams.num_input_layers)  # W1: MxD
W2 = np.random.randn(NNParams.num_output_layers, NNParams.num_hidden_layers) / np.sqrt(
    NNParams.num_hidden_layers)  # W2: KxM

# concat ones vector
W1 = concat_ones_vector(W1)  # W1: MxD+1
W2 = concat_ones_vector(W2)  # W2: KxM+1

# Do a gradient check first
# SKIP THIS PART FOR FASTER EXECUTION
'''
print('Running gradient check...')
ch = np.random.permutation(X_train.shape[0])
ch = ch[0:20]  # get the 20 first data
gradient_check(X_train[ch, :], t[ch, :], W1, W2)
'''

print('')

# define the learning rate based on the number of train data
NNParams.eta = 1 / len(X_train)

# train the Neural Network Model
W1, W2 = train(X_train, t, W1, W2, iterations=500, tol=1e-6, print_estimate=True)

# test the Neural Network Model
predicted = test(X_test, W1, W2)

# check predictions
wrong_counter = 0  # the number of wrong classifications made by the Neural Network

print('')
print('checking predictions...')
for i in range(len(predicted)):
    if predicted[i] == y_test_true[i]:
        print("data " + str(i) + ' classified as: ' + str(predicted[i]) + ' -> correct')
    elif predicted[i] != y_test_true[i]:
        print("data " + str(i) + ' classified as: ' + str(predicted[i]) + ' -> WRONG!')
        wrong_counter = wrong_counter + 1

print('')

# Accuracy

accuracy = ((len(X_test) - wrong_counter) / len(X_test)) * 100
print("accuracy: " + str(accuracy) + " %")
print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(X_test)) + ' images!')
