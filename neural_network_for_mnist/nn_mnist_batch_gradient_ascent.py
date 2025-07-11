# 1st Activation Function: tanh
# 2nd Activation Function: softmax
# Maximum Likelihood Estimate Function: Cross Entropy Function
# Train Algorithm: Batch Gradient Ascent
# Bias terms are used.

from read_mnist_data_from_files import *
from utilities import *


###############

class NNParams:
    num_input_nodes = 784  # D: number of nodes in the input layers (aka: num of features)
    num_hidden_nodes = 100  # M: number of nodes in the hidden layer
    num_output_nodes = 10  # K: number of nodes in the output layer (aka: num of categories)
    # Gradient ascent parameters
    eta = 0.1  # the learning rate for gradient ascent; it is modified according to the number of train data
    reg_lambda = 0.01  # the regularization parameter


###############

# FUNCTIONS #

# Feed-Forward
def forward(X, W1, W2):
    s1 = X.dot(W1.T)  # s1: NxM

    # activation function #1
    # o1 = h1(s1)  # o1: NxM
    # grad = h1_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM

    # activation function #2
    o1 = np.tanh(s1)  # o1: NxM
    grad = tanh_output_to_derivative(o1)  # the gradient of tanh function, grad: NxM

    # activation function #3
    # o1 = np.cos(s1)  # o1: NxM
    # grad = cos_output_to_derivative(o1)  # the gradient of cos function, grad: NxM

    o1 = concat_ones_vector(o1)  # o1: NxM+1
    s2 = o1.dot(W2.T)  # s2: NxK
    o2 = softmax(s2)  # o2: NxK
    return s1, o1, grad, s2, o2


# Helper function to evaluate the likelihood on the train dataset.
def likelihood(X, t, W1, W2):
    # num_examples = len(X_train)  # N: training set size

    # Feed-Forward to calculate our predictions
    _, _, _, s2, _ = forward(X, W1, W2)

    A = s2
    K = NNParams.num_output_nodes

    # Calculating the mle using the logsumexp trick
    maximum = np.max(A, axis=1)
    maximum = maximum.reshape(-1, 1)  # maximum: Nx1
    mle = np.sum(np.multiply(t, A)) - np.sum(maximum) \
          - np.sum(np.log(np.sum(np.exp(A - np.repeat(maximum, K, axis=1)), axis=1)))
    # ALTERNATIVE
    # mle = np.sum(np.multiply(t, np.log(o2)))

    mle *= 2  # for the gradient check to work

    # Add regularization term to likelihood (optional)
    mle -= NNParams.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return mle


def predict(X, W1, W2):
    # Feed-Forward
    _, _, _, _, o2 = forward(X, W1, W2)
    output = np.argmax(o2, axis=1)
    output = output.reshape(-1, 1)
    return output


# Train using Batch Gradient Ascent
# This function learns the parameter weights W1, W2 for the neural network and returns them.
# - iterations: Number of iterations through the training data for gradient ascent.
# - print_estimate: If True, print the estimate every 1000 iterations.
def train(X, t, W1, W2, iterations=500, tol=1e-6, print_estimate=False, X_val=None, y_val=None):
    # Run Batch Gradient Ascent
    lik_old = -np.inf
    for i in range(iterations):

        W1, W2, _, _ = grad_ascent(X, t, W1, W2)

        # Optionally print the estimate.
        # This is expensive because it uses the whole dataset.
        if print_estimate:
            lik = likelihood(X, t, W1, W2)
            if X_val is None or y_val is None:
                print("Iteration %i (out of %i), likelihood estimate: %f" % ((i + 1), iterations, float(lik)))
            else:
                # Print the estimate along with the accuracy on every epoch
                predicted = predict(X_val, W1, W2)
                err = np.not_equal(predicted, y_val)
                totalerrors = np.sum(err)
                acc = ((len(X_val) - totalerrors) / len(X_val)) * 100
                print("Iteration %i (out of %i), likelihood estimate: %f, accuracy on the validation set: %.2f %%"
                      % ((i + 1), iterations, float(lik), float(acc)))

            if np.abs(lik - lik_old) < tol:
                break
            lik_old = lik

    return W1, W2


# Update the Weight matrices using Gradient Ascent
def grad_ascent(X, t, W1, W2):
    # W1: MxD+1 = num_hidden_nodes X_train num_of_features
    # W2: KxM+1 = num_of_categories X_train num_hidden_nodes

    # Feed-Forward
    _, o1, grad, s2, o2 = forward(X, W1, W2)

    # Back-Propagation
    delta1 = t - o2  # delta1: 1xK
    W2_reduce = skip_first_column(W2)  # skip the first column of W2, so W2_reduce: KxM
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
            Ewplus = likelihood(X_train, t_train, W1, W2tmp)

            W2tmp = W2
            W2tmp[i, j] = W2[i, j] - epsilon
            Ewminus = likelihood(X_train, t_train, W1, W2tmp)

            numgradEw2[i, j] = (Ewplus - Ewminus) / (2 * epsilon)
    diff2 = np.sum(np.abs(gradEw2 - numgradEw2)) / np.sum(np.abs(gradEw2))
    print('The maximum absolute norm for parameter W2, in the gradient_check is: ' + str(diff2))


if __name__ == '__main__':
    mnist_dir = "mnisttxt"

    X_train, t_train = get_mnist_data(mnist_dir, 'train', one_hot=True)
    # y_train: the true categories vector for the train data
    y_train = np.argmax(t_train, axis=1)
    y_train = np.array(y_train).T

    print()

    X_test, t_test_true = get_mnist_data(mnist_dir, "test", one_hot=True)
    # y_test_true: the true categories vector for the test data
    y_test_true = np.argmax(t_test_true, axis=1)
    y_test_true = y_test_true.reshape(-1, 1)

    print()

    # normalize the data using range normalization
    X_train = X_train / 255
    X_test = X_test / 255

    # concat ones vector
    X_train = concat_ones_vector(X_train)
    X_test = concat_ones_vector(X_test)

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(NNParams.num_hidden_nodes, NNParams.num_input_nodes) / \
         np.sqrt(NNParams.num_input_nodes)  # W1: MxD
    W2 = np.random.randn(NNParams.num_output_nodes, NNParams.num_hidden_nodes) / \
         np.sqrt(NNParams.num_hidden_nodes)  # W2: KxM

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

    # define the learning rate based on the number of train data
    NNParams.eta = 0.5 / len(X_train)
    print('learning rate: ' + str(NNParams.eta))
    print()

    # train the Neural Network Model
    W1, W2 = train(
        X_train,
        t_train,
        W1,
        W2,
        iterations=500,
        tol=1e-6,
        print_estimate=True,
        X_val=X_test,
        y_val=y_test_true
    )

    # print the learned weights
    '''
    print('W1: ' + str(W1))
    print('W2: ' + str(W2))
    '''

    # test the Neural Network Model
    predicted = predict(X_test, W1, W2)

    # check predictions
    wrong_counter = 0  # the number of wrong classifications made by the NN

    true_positives = 0
    true_negatives = 0
    false_positives = 0  # the number of ham files classified as spam
    false_negatives = 0  # the number of spam files classified as ham

    print()
    print('checking predictions...')
    for i in range(len(predicted)):
        if predicted[i] == 1 and y_test_true[i] == 1:
            print("data" + str(i) + ' classified as: SPAM -> correct')
            true_positives = true_positives + 1
        elif predicted[i] == 1 and y_test_true[i] == 0:
            print("data" + str(i) + ' classified as: SPAM -> WRONG!')
            false_positives = false_positives + 1
            wrong_counter = wrong_counter + 1
        elif predicted[i] == 0 and y_test_true[i] == 0:
            print("data" + str(i) + ' classified as: HAM -> correct')
            true_negatives = true_negatives + 1
        elif predicted[i] == 0 and y_test_true[i] == 1:
            print("data" + str(i) + ' classified as: HAM -> WRONG!')
            false_negatives = false_negatives + 1
            wrong_counter = wrong_counter + 1

    print()

    # Accuracy

    accuracy = ((len(X_test) - wrong_counter) / len(X_test)) * 100
    print("accuracy: " + str(accuracy) + " %")
    print()

    # Calculate Precision-Recall

    print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(y_test_true.size) + ' files')
    print(
        "number of wrong spam classifications: " + str(false_positives) + ' out of ' + str(y_test_true.size) + ' files'
    )
    print(
        "number of wrong ham classifications: " + str(false_negatives) + ' out of ' + str(y_test_true.size) + ' files'
    )

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
