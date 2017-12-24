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
    return np.matrix(output)


def sigmoid_output_to_derivative(output):
    return np.multiply(output, (1-output))


def softmax(X):
    output = np.exp(X) / np.sum(np.exp(X), axis=1)
    return np.matrix(output)


def tanh_output_to_derivative(output):
    return 1 - np.square(output)
