import numpy as np
import pandas as pd
# import re
from pandas import DataFrame

# set options
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)

D = 784  # number of input layers (or number of pixels in the digit image)
K = 10  # number of output layers (or number of categories or number of digits)


###############


def read_data(path, testOrTrainFile):
    text_file = open(path + testOrTrainFile + '.txt', 'r')
    lines = text_file.readlines()
    text_file.close()

    X = [[0 for _ in range(D)] for _ in
         range(len(lines))]  # X: len(lines) X num_input_layers
    for i in range(len(lines)):
        tokens = lines[i].split(' ')
        for j in range(D):
            if j == D - 1:
                tokens[j] = tokens[j].replace('\n', '')
            X[i][j] = np.int(tokens[j])

    X = np.matrix(X)  # convert classification parameter to the appropriate data type
    return X


def get_mnist_data(mnist_dir, trainOrTest, one_hot=False):
    print('Reading ' + trainOrTest + ' files...')

    # read train images for digits 0,1, 2 and 3
    X = np.matrix  # 2D matrix
    y = np.matrix  # 1D matrix

    for i in range(K):
        print('Reading "' + trainOrTest + str(i) + '.txt"')
        X_class_i = read_data(mnist_dir, trainOrTest + str(i))
        N_train_i = X_class_i.shape[0]
        Y_class_i = np.repeat([i], N_train_i, axis=0)
        if i == 0:
            X = X_class_i
            y = Y_class_i
        else:
            X = np.concatenate((X, X_class_i), axis=0)
            y = np.concatenate((y, Y_class_i), axis=0)

    y = np.matrix(y).T

    if one_hot:
        # construct t_train: 1-hot matrix for the categories y_train
        t = np.zeros((y.shape[0], K))
        t[np.arange(y.shape[0]), y.ravel().tolist()] = 1

        return X, t
    else:
        return X, y


###############

# MAIN #

if __name__ == "__main__":
    mnist_dir = "./mnisttxt/"

    X_train, t_train = get_mnist_data(mnist_dir, 'train', one_hot=True)
    # y_train: the true categories vector for the train data
    y_train = np.argmax(t_train, axis=1)
    y_train = np.matrix(y_train).T

    print()
    print("X_train:")
    df = DataFrame(X_train)
    df.index = range(X_train.shape[0])
    df.columns = range(X_train.shape[1])
    print(df)

    print("y_train: " + str(y_train))

    print()

    X_test, t_test_true = get_mnist_data(mnist_dir, "test", one_hot=True)
    # y_test_true: the true categories vector for the test data
    y_test_true = np.argmax(t_test_true, axis=1)
    y_test_true = np.matrix(y_test_true).T

    print()
    print("X_test:")
    df = DataFrame(X_test)
    df.index = range(X_test.shape[0])
    df.columns = range(X_test.shape[1])
    print(df)

    print("y_test_true: " + str(y_test_true))

    print()
