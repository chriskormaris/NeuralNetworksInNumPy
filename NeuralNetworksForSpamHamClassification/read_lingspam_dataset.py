import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# I/O Libraries
from os import listdir
from os.path import isfile, join

__author__ = 'c.kormaris'


def generate_data(path, trainOrTest, feature_tokens, tfidf=False):
    if trainOrTest == 'train':
        spam_dir = path + "/spam-train/"
        ham_dir = path + "/nonspam-train/"
    else:
        spam_dir = path + "/spam-test/"
        ham_dir = path + "/nonspam-test/"

    spam_files = sorted([spam_dir + f for f in listdir(spam_dir) if isfile(join(spam_dir, f))])
    ham_files = sorted([ham_dir + f for f in listdir(ham_dir) if isfile(join(ham_dir, f))])
    files = list(spam_files)
    files.extend(ham_files)

    stop_words = set(stopwords.words('english'))

    if not tfidf:
        vectorizer = CountVectorizer(input='filename', stop_words=stop_words, vocabulary=feature_tokens)
    else:
        vectorizer = TfidfVectorizer(input='filename', stop_words=stop_words, vocabulary=feature_tokens)
    tfidf_data = vectorizer.fit_transform(files)
    # print(tfidf_data)

    x = tfidf_data.toarray()
    # print(x)
    with open('x_' + trainOrTest + '.txt', "wb") as f:
        if not tfidf:
            np.savetxt(f, x.astype(int), fmt='%i', delimiter=", ")
        else:
            np.savetxt(f, x.astype(int), fmt='%.4e', delimiter=", ")

    labels = [1] * len(spam_files)
    labels.extend([0] * len(ham_files))
    y = np.array(labels, dtype=np.int8)
    with open('y_' + trainOrTest + '.txt', "wb") as f:
        np.savetxt(f, y.astype(int), fmt='%i')

    return x, y


def get_classification_data(path, feature_tokens_dictionary_dir, construct_data=False, tfidf=False):

    feature_tokens = read_dictionary_file(feature_tokens_dictionary_dir)

    if construct_data:
        x_train, y_train = generate_data(path, 'train', feature_tokens, tfidf=tfidf)
        x_test, y_test = generate_data(path, 'test', feature_tokens, tfidf=tfidf)
    else:
        x_train = np.loadtxt('./x_train.txt', delimiter=', ')
        y_train = np.loadtxt('./y_train.txt', dtype=np.int8)
        x_test = np.loadtxt('./x_test.txt', delimiter=', ')
        y_test = np.loadtxt('./y_test.txt', dtype=np.int8)

    return x_train, y_train, x_test, y_test


def read_file(filename):
    with open(filename, "r") as f:
        return f.read()


# defines the label of the files based on their names
def read_labels(files):
    labels = []
    for file in files:
        if 'spam' in file:
            labels.append(1)
        elif 'ham' in file:
            labels.append(0)
    return labels


def read_dictionary_file(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines
