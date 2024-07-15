import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
import time
import os




# Loss function
def cross_entropy_loss(y, f_x):
    return -np.mean(y * np.log(f_x))


# logistic loss function
def binary_cross_entropy_loss(w, X, y, lambd, alpha):

    n=y.shape[0]
    z = np.dot(X,w)
    prediction = sigmoid(z)
    result = -(np.dot(y , np.log(prediction)) + np.dot(1-y , np.log(1-prediction)))/n

    return result              


def logistic_loss_gradient(w, X, Y, lambd, alpha, delta=1e-3):

    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  
    h = sigmoid(z)
    grad= X.T.dot(h-Y[:, np.newaxis])/n
    grad = grad + delta * w

    return  grad

def logistic_loss_hessian( w, X, Y, lambd, alpha, delta=1e-3):

    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=sigmoid(z)
    h= np.array(q*(1-sigmoid(z)))
    H = np.dot(np.transpose(X),h* X) / n
    H = H + delta * np.eye(d, d)

    return H


# Regularized loss function
def reg_loss(w, X, y, lambd, alpha):

    r = lambd * np.sum([alpha * wi**2 / (1 + alpha * wi**2) for wi in w])     
    reg_loss = binary_cross_entropy_loss(w, X, y, lambd, alpha) + r
    return reg_loss

def reg_loss_gradient(w, X, y, lambd, alpha, delta=1e-3):

    grad_r = 2 * lambd * alpha * w / (alpha * w**2 + 1)**2
    grad = logistic_loss_gradient(w, X, y, delta)+grad_r

    return grad

def reg_loss_hessian(w, X, y, lambd, alpha, delta=1e-3):

    diag = -4 * lambd * alpha * w * (alpha * w**2 - 1) / (alpha * w**2 + 1)**3
    reg_hessian = logistic_loss_hessian(w, X, y, delta) + np.diag(diag)
    
    return reg_hessian




# Sigmoid function
def softmax(z):
    z_exp = np.exp(z)
    softmax_probs = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    return softmax_probs

def sigmoid(z):
    return 1/(1+np.exp(-z))