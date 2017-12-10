# 1st Activation Function: tanh
# 2nd Activation Function: sigmoid
# Loss Function: Mean Squared Error Loss
# Train Algorithm: Stochastic Gradient Descent
# Bias terms are used.

# force the result of divisions to be float numbers
from __future__ import division

import numpy as np
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
    eta = 0.001  # the learning rate of gradient descent
    reg_lambda = 0.01  # the regularization parameter


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


def get_classification_data(spam_files_dir, ham_files_dir, files, labels, feature_tokens, trainOrTest):
    # classification parameter X_train
    X_2d_list = [[0 for _ in range(len(feature_tokens))] for _ in
                       range(len(files))]  # X_train: len(files) X_train len(feature_tokens)

    # reading files
    for i in range(len(files)):
        print("Reading " + trainOrTest + " file " + "'" + files[i] + "'" + "...")

        text = ''
        if labels[i] == 1:  # 1 is for class "SPAM"
            text = read_file(spam_files_dir + files[i])
        elif labels[i] == 0:  # 0 is for class "HAM"
            text = read_file(ham_files_dir + files[i])
        text_tokens = getTokens(text)

        # the feature vector contains features with Boolean values
        feature_vector = [0] * len(feature_tokens)
        for j in range(len(feature_tokens)):
            if text_tokens.__contains__(feature_tokens[j]):
                feature_vector[j] = 1
        feature_vector = tuple(feature_vector)

        X_2d_list[i][:] = feature_vector

    print('')

    # convert classification parameters to the appropriate data type
    X_train = np.array(X_2d_list)
    y_train = np.array(labels)

    return X_train, y_train


# extracts tokens from the given text
def getTokens(text):
    text_tokens = re.findall(r"[\w']+", text)
    # remove digits, special characters and convert to lowercase
    for k in range(len(text_tokens)):
        text_tokens[k] = text_tokens[k].lower()
        text_tokens[k] = text_tokens[k].replace("_", "")
        text_tokens[k] = re.sub("[0-9]+", "", text_tokens[k])
    text_tokens = set(text_tokens)  # remove duplicate tokens

    return text_tokens


# concat ones column vector as the first column of the matrix
def concat_ones_vector(X_train):
    ones_vector = np.ones((X_train.shape[0], 1))
    return np.concatenate((ones_vector, X_train), axis=1)


def tanh_output_to_derivative(output):
    return 1 - np.square(output)


def sigmoid(X_train):
    output = 1 / (1 + np.exp(-X_train))
    return np.matrix(output)


# Feed-Forward
def forward(X_train, W1, W2):
    s1 = X_train.dot(W1.t_train)  # s1: NxM
    o1 = np.tanh(s1)  # o1: NxM
    grad = tanh_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM
    o1 = concat_ones_vector(o1)  # o1: NxM+1
    s2 = o1.dot(W2.t_train)  # s2: NxK
    o2 = sigmoid(s2)  # o2: NxK
    return s1, o1, grad, s2, o2


# Helper function to evaluate the total loss of the dataset
def loss_function(X_train, t_train, W1, W2):
    num_examples = len(X_train)  # N: training set size

    # Feed-Forward to calculate our predictions
    _, _, _, _, o2 = forward(X_train, W1, W2)

    # Calculating the mean square error loss
    squared_error = np.square(o2 - t_train)
    data_loss = np.sum(squared_error) / 2

    # Add regularization term to loss (optional)
    data_loss += NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss / num_examples  # divide by number of examples and return


def test(X_train, W1, W2):
    # Feed-Forward
    _, _, _, _, o2 = forward(X_train, W1, W2)
    return np.argmax(o2, axis=1)


# This function learns the parameter weights W1, W2 for the neural network and returns them.
# - iterations: Number of iterations through the training data for gradient descent.
# - print_loss: If True, print the loss.
def train(X_train, t_train, W1, W2, epochs=50, tol=1e-6, print_loss=False):

    # Run Stochastic Gradient Descent
    num_examples = X_train.shape[0]
    s_old = -np.inf
    for e in range(epochs):

        s = 0
        for i in range(num_examples):
            xi = np.matrix(X_train[i, :])
            ti = np.matrix(t_train[i, :])
            W1, W2, _, _ = grad_descent(xi, ti, W1, W2)
            s = s + loss_function(xi, ti, W1, W2)

        # Optionally print the loss.
        if print_loss:
            print("Mean squared error loss after epoch %i: %f" % (e, loss_function(X_train, t_train, W1, W2)))

        if np.abs(s - s_old) <= tol:
            break

        s_old = s

    return W1, W2


# Update the Weight matrices using Gradient Descent
def grad_descent(X_train, t_train, W1, W2):
    # W1: MxD+1 = num_hidden_layers X_train num_of_features
    # W2: KxM+1 = num_of_categories X_train num_hidden_layers

    # Feed-Forward
    _, o1, grad, _, o2 = forward(X_train, W1, W2)

    # Back-Propagation
    delta1 = o2 - t_train  # delta1: NxK
    W2_reduce = W2[np.ix_(np.arange(W2.shape[0]), np.arange(1, W2.shape[1]))]  # skip the first column of W2: KxM
    delta2 = np.dot(delta1, W2_reduce)  # delta2: NxM
    delta3 = np.multiply(delta2, grad)  # element-wise multiplication, delta3: NxM

    dW1 = np.dot(delta3.t_train, X_train)  # MxD+1
    dW2 = np.dot(delta1.t_train, o1)  # KxM+1

    # Add regularization terms
    dW1 = dW1 + NNParams.reg_lambda * W1
    dW2 = dW2 + NNParams.reg_lambda * W2

    # Update gradient descent parameters
    W1 = W1 - NNParams.eta * dW1
    W2 = W2 - NNParams.eta * dW2

    return W1, W2, dW1, dW2


def gradient_check(X_train, t_train, W1, W2):
    _, _, gradEw1, gradEw2 = grad_descent(X_train, t_train, W1, W2)
    epsilon = 1e-6

    # gradient_check for parameter W1
    numgradEw1 = np.zeros(W1.shape)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1tmp = W1
            W1tmp[i, j] = W1[i, j] + epsilon
            Ewplus = loss_function(X_train, t_train, W1tmp, W2)

            W1tmp = W1
            W1tmp[i, j] = W1[i, j] - epsilon
            Ewminus = loss_function(X_train, t_train, W1tmp, W2)

            numgradEw1[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff1 = np.linalg.norm(gradEw1 - numgradEw1) / np.linalg.norm(gradEw1 + numgradEw1)
    print('The maximum absolute norm for parameter W1, in the gradient_check is: ' + str(diff1))

    # gradient_check for parameter W2
    numgradEw2 = np.zeros(W2.shape)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2tmp = W2
            W2tmp[i, j] = W2[i, j] + epsilon
            Ewplus = loss_function(X_train, t_train, W1, W2tmp)

            W2tmp = W2
            W2tmp[i, j] = W2[i, j] - epsilon
            Ewminus = loss_function(X_train, t_train, W1, W2tmp)

            numgradEw2[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff2 = np.linalg.norm(gradEw2 - numgradEw2) / np.linalg.norm(gradEw2 + numgradEw2)
    print('The maximum absolute norm for parameter W2, in the gradient_check is: ' + str(diff2))


###############

# MAIN #

feature_dictionary_dir = "feature_dictionary.txt"

spam_train_dir = "LingspamDataset/spam-train/"
ham_train_dir = "LingspamDataset/nonspam-train/"
spam_test_dir = "LingspamDataset/spam-test/"
ham_test_dir = "LingspamDataset/nonspam-test/"

# read feature dictionary from file
feature_tokens = read_dictionary_file(feature_dictionary_dir)
NNParams.num_input_layers = len(feature_tokens)

print("Reading TRAIN files...")
spam_train_files = sorted([f for f in listdir(spam_train_dir) if isfile(join(spam_train_dir, f))])
ham_train_files = sorted([f for f in listdir(ham_train_dir) if isfile(join(ham_train_dir, f))])
train_files = list(spam_train_files)
train_files.extend(ham_train_files)
train_labels = [1] * len(spam_train_files)
train_labels.extend([0] * len(ham_train_files))
X_train, y_train = get_classification_data(spam_train_dir, ham_train_dir, train_files, train_labels, feature_tokens, 'train')

print('')

print("Reading TEST files...")
spam_test_files = sorted([f for f in listdir(spam_test_dir) if isfile(join(spam_test_dir, f))])
ham_test_files = sorted([f for f in listdir(ham_test_dir) if isfile(join(ham_test_dir, f))])
test_files = list(spam_test_files)
test_files.extend(ham_test_files)
test_true_labels = [1] * len(spam_test_files)
test_true_labels.extend([0] * len(ham_test_files))
X_test, y_test_true = get_classification_data(spam_test_dir, ham_test_dir, test_files, test_true_labels, feature_tokens, 'test')

print('')

# normalize the data using mean normalization
X_train = X_train - np.mean(X_train)
X_test = X_test - np.mean(X_test)

# concat ones vector
X_train = concat_ones_vector(X_train)
X_test = concat_ones_vector(X_test)

# t_train: 1-hot matrix for the categories y_train
t_train = np.zeros((y_train.shape[0], NNParams.num_output_layers))
t_train[np.arange(y_train.shape[0]), y_train] = 1

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
print('Running gradient check...')
ch = np.random.permutation(X_train.shape[0])
ch = ch[0:20]  # get the 20 first data
gradient_check(X_train[ch, :], t_train[ch, :], W1, W2)

print('')

# train the Neural Network Model
W1, W2 = train(X_train, t_train, W1, W2, epochs=50, tol=1e-6, print_loss=True)

# test the Neural Network Model
predicted = test(X_test, W1, W2)


# check predictions
wrong_counter = 0  # the number of wrong classifications made by Logistic Regression
spam_counter = 0  # the number of spam files
ham_counter = 0  # the number of ham files
wrong_spam_counter = 0  # the number of spam files classified as ham
wrong_ham_counter = 0  # the number of ham files classified as spam

print('')
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

print('')

# Accuracy

accuracy = ((len(X_test) - wrong_counter) / len(X_test)) * 100
print("accuracy: " + str(accuracy) + " %")
print('')

# Calculate Precision-Recall

print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(X_test)) + ' files')
print("number of wrong spam classifications: " + str(wrong_spam_counter) + ' out of ' + str(spam_counter) + ' spam files')
print("number of wrong ham classifications: " + str(wrong_ham_counter) + ' out of ' + str(ham_counter) + ' ham files')

print('')

spam_precision = (spam_counter - wrong_spam_counter) / (spam_counter - wrong_spam_counter + wrong_ham_counter)
print("precision for spam files: " + str(spam_precision))
ham_precision = (ham_counter - wrong_ham_counter) / (ham_counter - wrong_ham_counter + wrong_spam_counter)
print("precision for ham files: " + str(ham_precision))

spam_recall = (spam_counter - wrong_spam_counter) / spam_counter
print("recall for spam files: " + str(spam_recall))
ham_recall = (ham_counter - wrong_ham_counter) / ham_counter
print("recall for ham files: " + str(ham_recall))
