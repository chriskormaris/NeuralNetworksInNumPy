import numpy as np
import re

__author__ = 'c.kormaris'


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
