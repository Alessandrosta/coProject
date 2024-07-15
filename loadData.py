import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
import time
import os


# Dataset function
def load_mnist_libsvm():
    # Load the dataset using load_svmlight_file
    X_train, y_train = load_svmlight_file('mnist')
    X_test, y_test = load_svmlight_file('mnist.t')

    # Convert to dense arrays
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Ensure that both training and test sets have the same number of features
    n_features = X_train.shape[1]
    if X_test.shape[1] != n_features:
        X_test = np.hstack([X_test, np.zeros((X_test.shape[0], n_features - X_test.shape[1]))])

    # One-hot encode the labels
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train).toarray()
    y_test = encoder.transform(y_test).toarray()

    return X_train, X_test, y_train, y_test

def load_a9a():
    current_file_path = os.path.realpath(__file__)
    current_file_directory = os.path.dirname(current_file_path)

    # Use os.path.join for proper path construction
    train_file_path = os.path.join(current_file_directory, 'Data', 'a9a.txt')
    test_file_path = os.path.join(current_file_directory, 'Data', 'a9a.t')

    # Load the training and test data
    X_train, y_train = load_svmlight_file(train_file_path)
    X_test, y_test = load_svmlight_file(test_file_path)

    X_train = X_train.toarray()
    y_train= [0 if e == -1 else e for e in y_train]
    y_train=np.array(y_train)

    X_test = X_test.toarray()
    y_test= [0 if e == -1 else e for e in y_test]
    y_test=np.array(y_test)
    return X_train, X_test, y_train, y_test


def load_ijcnn1():
    current_file_path = os.path.realpath(__file__)
    current_file_directory = os.path.dirname(current_file_path)

    # Use os.path.join for proper path construction
    train_file_path = os.path.join(current_file_directory, 'Data', 'ijcnn1')
    test_file_path = os.path.join(current_file_directory, 'Data', 'ijcnn1.t')

    # Load the training and test data
    X_train, y_train = load_svmlight_file(train_file_path)
    X_test, y_test = load_svmlight_file(test_file_path)
    
    # Convert the sparse matrix to a dense array
    X_train = X_train.toarray()
    y_train = np.array([0 if e == -1 else e for e in y_train])

    X_test = X_test.toarray()
    y_test = np.array([0 if e == -1 else e for e in y_test])
    
    return X_train, X_test, y_train, y_test