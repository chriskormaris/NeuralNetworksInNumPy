# This is the same NEURAL NETWORK as in Exercise 9 (SLIDE 31), but with the rightmost SIGMOID. #
# 1st Activation Function: tanh
# 2nd Activation Function: sigmoid
# Cost Function: Mean Squared Error cost
# Train Algorithm: Mini-batch Gradient Descent
# Bias terms are used.

from read_lingspam_dataset import *
from Utilities import *

import numpy as np

__author__ = 'c.kormaris'

feature_dictionary_dir = "./feature_dictionary.txt"
path = "./LingspamDataset"

###############


class NNParams:
    num_input_units = 1000  # D: number of nodes in the input layers (aka: no of features)
    num_hidden_units = 50  # M: number of nodes in the hidden layer
    num_output_units = 2  # K: number of nodes in the output layer (aka: no of categories)
    # Gradient descent parameters
    eta = 0.001  # the learning rate of gradient descent
    reg_lambda = 0.01  # the regularization parameter
    batch_size = 50
    epochs = 50
    tol = 1e-6

###############

# FUNCTIONS #

# Feed-Forward
def forward(X, W1, W2):
    s1 = X.dot(W1.T)  # s1: NxM
    o1 = np.tanh(s1)  # o1: NxM
    grad = tanh_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM
    o1 = concat_ones_vector(o1)  # o1: NxM+1
    s2 = o1.dot(W2.T)  # s2: NxK
    o2 = sigmoid(s2)  # o2: NxK
    return s1, o1, grad, s2, o2


# Helper function to evaluate the total cost of the dataset
def cost_function(X, t, W1, W2):
    num_examples = len(X)  # N: training set size

    # Feed-Forward to calculate our predictions
    _, _, _, _, o2 = forward(X, W1, W2)

    # Calculating the mean square error cost
    squared_error = np.square(o2 - t)
    data_cost = np.sum(squared_error) / 2

    # Add regularization term to cost (optional)
    data_cost += NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_cost / num_examples  # divide by number of examples and return


def test(X, W1, W2):
    # Feed-Forward
    _, _, _, _, o2 = forward(X, W1, W2)
    return np.argmax(o2, axis=1)


# This function learns the parameter weights W1, W2 for the neural network and returns them.
# - iterations: Number of iterations through the training data for gradient descent.
# - print_cost_function: If True, print the cost.
def train(X, t, W1, W2, epochs=50, tol=1e-6, print_cost_function=False):

    # Run Mini-batch Gradient Descent
    num_examples = X.shape[0]
    s_old = -np.inf
    for e in range(epochs):

        s = 0
        iterations = int(np.ceil(num_examples / NNParams.batch_size))
        for i in range(iterations):
            start_index = int(i * NNParams.batch_size)
            end_index = int(i * NNParams.batch_size + NNParams.batch_size)
            W1, W2, _, _ = gradient_descent(np.array(X[start_index:end_index, :]), np.array(t[start_index:end_index, :]), W1, W2)
            s = s + cost_function(np.array(X[start_index:end_index, :]), np.array(t[start_index:end_index, :]), W1, W2)

        # Optionally print the cost.
        if print_cost_function:
            print("Mean squared error cost function after epoch %i: %f" % (e, cost_function(X, t, W1, W2)))

        if np.abs(s - s_old) < tol:
            break

        s_old = s

    return W1, W2


# Update the Weight matrices using Gradient Descent
def gradient_descent(X, t, W1, W2):
    # W1: MxD+1 = num_hidden_layers X_train num_of_features
    # W2: KxM+1 = num_of_categories X_train num_hidden_layers

    # Feed-Forward
    _, o1, grad, _, o2 = forward(X, W1, W2)

    # Back-Propagation

    # sum1 = np.array(np.sum(t_train, axis=1)).T  # sum1: Nx1
    # t_train = np.matlib.repmat(sum1, 1, K)  # t_train: NxK, each row contains the same sum values in each column
    # delta1 = np.multiply(o2, t_train) - t_train  # delta1: NxK
    delta1 = o2 - t  # delta1: NxK, since t_train is one-hot matrix, then t_train=1, so we can omit it

    W2_reduce = skip_first_column(W2)  # skip the first column of W2: KxM
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

    return W1, W2, dW1, dW2


# IT DOES NOT WORK CORRECTLY YET!
def gradient_check(X, t, W1, W2):
    _, _, gradEw1, gradEw2 = gradient_descent(X, t, W1, W2)
    epsilon = 1e-6

    # gradient_check for parameter W1
    numgradEw1 = np.zeros(W1.shape)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1tmp = W1
            W1tmp[i, j] = W1[i, j] + epsilon
            Ewplus = cost_function(X, t, W1tmp, W2)

            W1tmp = W1
            W1tmp[i, j] = W1[i, j] - epsilon
            Ewminus = cost_function(X, t, W1tmp, W2)

            numgradEw1[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff1 = np.linalg.norm(gradEw1 - numgradEw1) / np.linalg.norm(gradEw1 + numgradEw1)
    print('The maximum absolute norm for parameter W1, in the gradient_check is: ' + str(diff1))

    # gradient_check for parameter W2
    numgradEw2 = np.zeros(W2.shape)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2tmp = W2
            W2tmp[i, j] = W2[i, j] + epsilon
            Ewplus = cost_function(X, t, W1, W2tmp)

            W2tmp = W2
            W2tmp[i, j] = W2[i, j] - epsilon
            Ewminus = cost_function(X, t, W1, W2tmp)

            numgradEw2[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff2 = np.linalg.norm(gradEw2 - numgradEw2) / np.linalg.norm(gradEw2 + numgradEw2)
    print('The maximum absolute norm for parameter W2, in the gradient_check is: ' + str(diff2))


###############

# MAIN #

if __name__ == '__main__':

    # read feature dictionary from file
    feature_tokens = read_dictionary_file(feature_dictionary_dir)
    NNParams.num_input_units = len(feature_tokens)

    print("Getting train and test data...")
    X_train, y_train, X_test, y_test = get_classification_data(path, feature_dictionary_dir)

    print()
    
    # normalize the data using mean normalization
    X_train = X_train - np.mean(X_train)
    X_test = X_test - np.mean(X_test)
    
    # concat ones vector
    X_train = concat_ones_vector(X_train)
    X_test = concat_ones_vector(X_test)
    
    # t_train: 1-hot matrix for the categories y_train
    t_train = np.zeros((y_train.shape[0], NNParams.num_output_units))
    t_train[np.arange(y_train.shape[0]), y_train] = 1
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(NNParams.num_hidden_units, NNParams.num_input_units) / np.sqrt(
        NNParams.num_input_units)  # W1: MxD
    W2 = np.random.randn(NNParams.num_output_units, NNParams.num_hidden_units) / np.sqrt(
        NNParams.num_hidden_units)  # W2: KxM
    
    # concat ones vector
    W1 = concat_ones_vector(W1)  # W1: MxD+1
    W2 = concat_ones_vector(W2)  # W2: KxM+1
    
    # Do a gradient check first
    # SKIP THIS PART FOR FASTER EXECUTION
    '''
    print('Running gradient check...')
    ch = np.random.permutation(X_train.shape[0])
    ch = ch[0:20]  # get the 20 first data
    gradient_check(X_train[ch, :], t_train[ch, :], W1, W2)
    '''
    
    print()
    
    # train the Neural Network Model
    W1, W2 = train(X_train, t_train, W1, W2, epochs=NNParams.epochs, tol=NNParams.tol, print_cost_function=True)
    
    # test the Neural Network Model
    y_test_predicted = test(X_test, W1, W2)
    
    # check predictions
    wrong_counter = 0  # the number of wrong classifications made by the NN
    false_positives = 0  # the number of ham files classified as spam
    false_negatives = 0  # the number of spam files classified as ham
    true_positives = 0
    true_negatives = 0

    print()
    print('checking predictions...')
    for i in range(len(y_test_predicted)):
        if y_test_predicted[i] == 1 and y_test[i] == 1:
            print("data" + str(i) + ' classified as: SPAM -> correct')
            true_positives = true_positives + 1
        elif y_test_predicted[i] == 1 and y_test[i] == 0:
            print("data" + str(i) + ' classified as: SPAM -> WRONG!')
            false_positives = false_positives + 1
            wrong_counter = wrong_counter + 1
        elif y_test_predicted[i] == 0 and y_test[i] == 1:
            print("data" + str(i) + ' classified as: HAM -> WRONG!')
            false_negatives = false_negatives + 1
            wrong_counter = wrong_counter + 1
        elif y_test_predicted[i] == 0 and y_test[i] == 0:
            print("data" + str(i) + ' classified as: HAM -> correct')
            true_negatives = true_negatives + 1

    print()

    # Accuracy

    accuracy = ((len(X_test) - wrong_counter) / len(X_test)) * 100
    print("accuracy: " + str(accuracy) + " %")
    print()

    # Calculate Precision-Recall

    print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(y_test.size) + ' files')
    print("number of wrong spam classifications: " + str(false_positives) + ' out of ' + str(y_test.size) + ' files')
    print("number of wrong ham classifications: " + str(false_negatives) + ' out of ' + str(y_test.size) + ' files')

    print()

    spam_precision = true_positives / (true_positives + false_positives) * 100
    print("precision for spam files: " + str(spam_precision) + " %")
    ham_precision = true_negatives / (true_negatives + false_negatives) * 100
    print("precision for ham files: " + str(ham_precision) + " %")

    spam_recall = true_positives / (true_positives + false_negatives) * 100
    print("recall for spam files: " + str(spam_recall) + " %")
    ham_recall = true_negatives / (true_negatives + false_positives) * 100
    print("recall for ham files: " + str(ham_recall) + " %")

    spam_f1_score = 2 * spam_precision * spam_recall / (spam_precision + spam_recall)
    print("f1-score for spam files: " + str(spam_f1_score) + " %")
    ham_f1_score = 2 * ham_precision * ham_recall / (ham_precision + ham_recall)
    print("f1-score for ham files: " + str(ham_f1_score) + " %")
