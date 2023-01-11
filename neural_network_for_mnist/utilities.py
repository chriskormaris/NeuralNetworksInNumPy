import numpy as np

# ignore errors
np.seterr(all='ignore')


#####

# HELPER FUNCTIONS #

# activation function #1
def h1(X):
    return np.log(1 + np.exp(X))


# activation function #1 derivative / the same as the sigmoid function
def h1_output_to_derivative(output):
    return sigmoid(output)


# activation function #2 derivative
def tanh_output_to_derivative(output):
    return 1 - np.square(output)


# activation function #3 derivative
def cos_output_to_derivative(output):
    return -np.sin(output)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


# activation function for the 2nd layer
def softmax(X):
    denominator = np.sum(np.exp(X), axis=1)
    return np.exp(X) / denominator.reshape((denominator.shape[0], 1))


# concat ones column vector as the first column of the matrix (adds bias term)
def concat_ones_vector(X):
    ones_vector = np.ones((X.shape[0], 1))
    return np.concatenate((ones_vector, X), axis=1)
