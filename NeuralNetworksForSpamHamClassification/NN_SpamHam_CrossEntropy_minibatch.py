# This is the same NEURAL NETWORK as in Exercise 10 (SLIDE 35 with a leftmost sigmoid function instead of tanh) #
# 1st Activation Function: sigmoid
# 2nd Activation Function: softmax
# Loss Function: Cross Entropy Loss

# force the result of divisions to be float numbers
from __future__ import division

import numpy as np
import numpy.matlib
import re

# I/O Libraries
from os import listdir
from os.path import isfile, join

__author__ = 'c.kormaris'

np.seterr(all='ignore')


###############


class NNParams:
    num_input_layers = 1000  # D: number of nodes in the input layers (aka: no of features)
    num_hidden_layers = 3  # M: number of nodes in the hidden layer
    num_output_layers = 2  # K: number of nodes in the output layer (aka: no of categories)
    # Gradient descent parameters
    eta = 0.01  # the learning rate of gradient descent
    reg_lambda = 0.01  # the regularization parameter
    batch_size = 200


###############

# FUNCTIONS #


def read_dictionary_file(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines


def read_file(filename):
    text_file = open(filename, "r")
    text = text_file.read()
    return text


# defines the label of the files based on their names
def read_labels(files):
    labels = []
    for file in files:
        if "spam" in str(file):
            labels.append(1)
        elif "ham" in str(file):
            labels.append(0)
    return labels


def get_classification_data(files_dir, files, labels, feature_tokens, trainOrTest):
    # classification parameter X
    X_2d_list = [[0 for _ in range(len(feature_tokens))] for _ in
                       range(len(files))]  # X: len(files) x len(feature_tokens)

    # reading files
    for i in range(len(files)):
        print("Reading " + trainOrTest + " file " + "'" + files[i] + "'" + "...")

        text = read_file(files_dir + files[i])

        text_tokens = re.findall(r"[\w']+", text)

        # remove digits, special characters and convert to lowercase
        for k in range(len(text_tokens)):
            text_tokens[k] = text_tokens[k].lower()
            text_tokens[k] = text_tokens[k].replace("_", "")
            text_tokens[k] = re.sub("[0-9]+", "", text_tokens[k])

        text_tokens = set(text_tokens)  # remove duplicate tokens

        # the feature vector contains features with Boolean values
        feature_vector = [0] * len(feature_tokens)
        for j in range(len(feature_tokens)):
            if text_tokens.__contains__(feature_tokens[j]):
                feature_vector[j] = 1
        feature_vector = tuple(feature_vector)

        X_2d_list[i][:] = feature_vector

    print("\n")

    # convert classification parameters to the appropriate data type
    X = np.array(X_2d_list)
    y = np.array(labels)

    return X, y


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return np.matrix(output)


def sigmoid_output_to_derivative(output):
    return np.multiply(output, (1-output))


def softmax(x):
    output = np.exp(x) / np.sum(np.exp(x), axis=1)
    return np.matrix(output)


# concat ones column vector as the first column of the matrix
def concat_ones_vector(x):
    ones_vector = np.ones((x.shape[0], 1))
    return np.concatenate((ones_vector, x), axis=1)


# Forward propagation
def forward(X, W1, W2):
    s1 = X.dot(W1.T)  # s1: NxM
    o1 = sigmoid(s1)  # o1: NxM
    grad = sigmoid_output_to_derivative(o1)  # the gradient of sigmoid function, grad: NxM
    o1 = concat_ones_vector(o1)  # o1: NxM+1
    s2 = o1.dot(W2.T)  # s2: NxK
    o2 = softmax(s2)  # o2: NxK
    return s1, o1, grad, s2, o2


# Helper function to evaluate the total loss of the dataset
def loss_function(X, t, W1, W2):
    num_examples = len(X)  # N: training set size

    # Forward propagation to calculate our predictions
    _, _, _, _, o2 = forward(X, W1, W2)

    # Calculating the loss
    logprobs = -np.multiply(t, np.log(o2))
    data_loss = np.sum(logprobs)  # cross entropy loss

    # Add regularization term to loss (optional)
    data_loss += NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss


def test(X, W1, W2):
    # Forward propagation
    _, _, _, _, o2 = forward(X, W1, W2)
    return np.argmax(o2, axis=1)


# This function learns the parameter weights W1, W2 for the neural network and returns them.
# - iterations: Number of iterations through the training data for gradient descent.
# - print_loss: If True, print the loss.
def train(X, y, epochs=50, tol=1e-6, print_loss=False):
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

    # Run Mini-batch Gradient Descent
    num_examples = X.shape[0]
    s_old = -np.inf
    for e in range(epochs):

        s = 0
        iterations = int(np.ceil(num_examples / NNParams.batch_size))
        for i in range(iterations):
            start_index = int(i * NNParams.batch_size)
            end_index = int(i * NNParams.batch_size + NNParams.batch_size - 1)
            W1, W2 = grad_descent(np.matrix(X[start_index:end_index, :]), np.matrix(t[start_index:end_index, :]), W1, W2)
            s = s + loss_function(np.matrix(X[start_index:end_index, :]), np.matrix(t[start_index:end_index, :]), W1, W2)

            # Optionally print the loss.
        if print_loss:
            print("Cross entropy loss after epoch %i: %f" % (e, loss_function(X, t, W1, W2)))

        if np.abs(s - s_old) < tol:
            break

        s_old = s

    return W1, W2


# Update the Weight matrices using Gradient Descent
def grad_descent(X, t, W1, W2):
    K = NNParams.num_output_layers
    # W1: MxD+1 = num_hidden_layers x num_of_features
    # W2: KxM+1 = num_of_categories x num_hidden_layers

    # Forward propagation
    _, o1, grad, _, o2 = forward(X, W1, W2)

    # Backpropagation

    #sum1 = np.matrix(np.sum(t, axis=1)).T  # sum1: Nx1
    #T = np.matlib.repmat(sum1, 1, K)  # T: NxK, each row contains the same sum values in each column
    #delta1 = np.multiply(o2, T) - t  # delta1: NxK
    delta1 = o2 - t  # delta1: NxK, since t is one-hot matrix, then T=1, so we can omit it

    W2_reduce = W2[np.ix_(np.arange(W2.shape[0]), np.arange(1, W2.shape[1]))]  # skip the first column of W2: KxM
    delta2 = np.dot(delta1, W2_reduce)  # delta2: NxM
    delta3 = np.multiply(delta2, grad)  # element-wise multiplication, delta3: NxM

    dW1 = np.dot(delta3.T, X)  # MxD+1
    dW2 = np.dot(delta1.T, o1)  # KxM+1

    # Add regularization terms
    dW1 = dW1 + NNParams.reg_lambda * W1
    dW2 = dW2 + NNParams.reg_lambda * W2

    # Update gradient descent parameters
    W1 = W1 - NNParams.eta * dW1
    W2 = W2 - NNParams.eta * dW2

    return W1, W2


###############

# MAIN #

train_dir = "TRAIN/"
test_dir = "TEST/"
feature_dictionary_dir = "feature_dictionary.txt"

# read feature dictionary from file
feature_tokens = read_dictionary_file(feature_dictionary_dir)
NNParams.num_input_layers = len(feature_tokens)

print("Reading TRAIN files...")
train_files = sorted([f for f in listdir(train_dir) if isfile(join(train_dir, f))])
train_labels = read_labels(train_files)
X_train, y_train = get_classification_data(train_dir, train_files, train_labels, feature_tokens, 'train')

print("\n")

print("Reading TEST files...")
test_files = sorted([f for f in listdir(test_dir) if isfile(join(test_dir, f))])
test_labels = read_labels(test_files)
X_test, y_test_true = get_classification_data(test_dir, test_files, test_labels, feature_tokens, 'test')

print("\n")

# concat ones vector
X_train = concat_ones_vector(X_train)
X_test = concat_ones_vector(X_test)

# normalize the data using mean normalization
X_train = X_train - np.mean(np.mean(X_train))
X_test = X_test - np.mean(np.mean(X_test))

# train the Neural Network Model
W1, W2 = train(X_train, y_train, epochs=50, tol=1e-6, print_loss=True)

# test the Neural Network Model
predicted = test(X_test, W1, W2)


# check predictions
wrong_counter = 0  # the number of wrong classifications made by Logistic Regression
spam_counter = 0  # the number of spam files
ham_counter = 0  # the number of ham files
wrong_spam_counter = 0  # the number of spam files classified as ham
wrong_ham_counter = 0  # the number of ham files classified as spam

print("\n")
print('checking predictions...')
for i in range(len(predicted)):
    if predicted[i] == 1 and y_test_true[i] == 1:
        print("data" + str(i) + ' classified as: SPAM -> correct')
        spam_counter = spam_counter + 1
    elif predicted[i] == 1 and y_test_true[i] == 0:
        print("data" + str(i) + ' classified as: SPAM -> WRONG!')
        ham_counter = ham_counter + 1
        wrong_ham_counter = wrong_ham_counter + 1
        wrong_counter = wrong_counter + 1
    elif predicted[i] == 0 and y_test_true[i] == 1:
        print("data" + str(i) + ' classified as: HAM -> WRONG!')
        spam_counter = spam_counter + 1
        wrong_spam_counter = wrong_spam_counter + 1
        wrong_counter = wrong_counter + 1
    elif predicted[i] == 0 and y_test_true[i] == 0:
        print("data" + str(i) + ' classified as: HAM -> correct')
        ham_counter = ham_counter + 1

print('\n')

# Accuracy

accuracy = ((len(X_test) - wrong_counter) / len(X_test)) * 100
print("accuracy: " + str(accuracy) + " %")
print("\n")

# Calculate Precision-Recall

print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(X_test)) + ' files')
print("number of wrong spam classifications: " + str(wrong_spam_counter) + ' out of ' + str(spam_counter) + ' spam files')
print("number of wrong ham classifications: " + str(wrong_ham_counter) + ' out of ' + str(ham_counter) + ' ham files')

print("\n")

spam_precision = (spam_counter - wrong_spam_counter) / (spam_counter - wrong_spam_counter + wrong_ham_counter)
print("precision for spam files: " + str(spam_precision))
ham_precision = (ham_counter - wrong_ham_counter) / (ham_counter - wrong_ham_counter + wrong_spam_counter)
print("precision for ham files: " + str(ham_precision))

spam_recall = (spam_counter - wrong_spam_counter) / (spam_counter)
print("recall for spam files: " + str(spam_recall))
ham_recall = (ham_counter - wrong_ham_counter) / (ham_counter)
print("recall for ham files: " + str(ham_recall))
