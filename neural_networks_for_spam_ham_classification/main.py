from read_lingspam_dataset import *


if __name__ == '__main__':
    feature_dictionary_dir = "feature_dictionary.txt"
    path = "LingspamDataset"

    # get train and test data
    print("Reading train and test files...")
    X_train, y_train, X_test, y_test = get_classification_data(
        path,
        feature_dictionary_dir,
        construct_data=True,
        tfidf=False
    )

    print("X_train length: ")
    print(X_train.shape)
    print()

    print("y_train length: ")
    print(y_train.shape)
    print()

    print("X_test length: ")
    print(X_test.shape)
    print()

    print("y_test length: ")
    print(y_test.shape)
    print()
