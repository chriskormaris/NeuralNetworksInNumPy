import numpy as np

# ignore errors
np.seterr(all='ignore')

__author__ = 'c.kormaris'


###############

# HELPER FUNCTIONS #

# concat ones column vector as the first column of the matrix
def concat_ones_vector(X):
    ones_vector = np.ones((X.shape[0], 1))
    return np.concatenate((ones_vector, X), axis=1)


def sigmoid(X):
    output = 1 / (1 + np.exp(-X))
    return np.array(output)


def sigmoid_output_to_derivative(output):
    return np.multiply(output, (1-output))


def softmax(X):
    denominator = np.sum(np.exp(X), axis=1)
    denominator = np.reshape(denominator, (X.shape[0], 1))
    output = np.exp(X) / denominator
    return np.array(output)


def tanh_output_to_derivative(output):
    return 1 - np.square(output)


def skip_first_column(X):
    return X[:, 1:]


def read_dictionary_file(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines
