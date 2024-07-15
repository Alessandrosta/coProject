import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
import time
import os
import mathFunctions


def stochastic_newton(X,y, weights, learning_rate, minibatch_size, hessian_function, gradient_function, Loss_function, M, lambd, alpha):
    # choose batch
    if minibatch_size >= X.shape[0]:
        X_batch = X
        y_batch = y
    else:
        batch =  np.random.choice(X.shape[0], minibatch_size, replace=False)
        X_batch = X[batch]
        y_batch = y[batch]
    
    # calculate xk+1
    Hk = hessian_function(weights, X_batch, y_batch, lambd, alpha)
    gk = gradient_function(weights, X_batch, y_batch, lambd, alpha)

    loss = 1000
    #update only if successful
    old_loss = Loss_function(weights, X, y, lambd, alpha)
    new_weights = weights - learning_rate*np.dot(np.linalg.inv(Hk), gk)
    new_loss = Loss_function(new_weights, X, y, lambd, alpha)

    if old_loss>new_loss:
        weights = new_weights
        loss = new_loss
    else:
        loss = old_loss


    return weights, loss





def stochastic_cubic_newton(X,y, weights, learning_rate, minibatch_size, hessian_function, gradient_function, Loss_function, M, lambd, alpha):
    
    
    if minibatch_size >= X.shape[0]:
        X_batch = X
        y_batch = y
    else:
        batch =  np.random.choice(X.shape[0], minibatch_size, replace=False)
        X_batch = X[batch]
        y_batch = y[batch]    

    # calculate xk+1
    Hk = hessian_function(weights, X_batch, y_batch, lambd, alpha)
    gk = gradient_function(weights, X_batch, y_batch, lambd, alpha)
    

    loss = 1000
    # update only if successful
    old_prediction = mathFunctions.sigmoid(np.dot(X, weights))
    old_loss = Loss_function(weights, X, y, lambd, alpha)
    new_weights = weights + cubic_regularization_subproblem_gd(gk, Hk, M, learning_rate=0.1, iterations=100)
    new_prediction = mathFunctions.sigmoid(np.dot(X, new_weights))
    new_loss = Loss_function(new_weights, X, y, lambd, alpha)
    #weights += cubic_regularization_subproblem_gd(gk, Hk, M, learning_rate=0.1, iterations=100)
    #pred = sigmoid(np.dot(X, weights))
    #loss = Loss_function(y, pred)


    if old_loss>new_loss:
        weights = new_weights
        loss = new_loss
    else:
        loss = old_loss
#
    return weights, loss



def cubic_regularization_subproblem_gd(g, H, M, learning_rate, iterations):
    w = np.zeros_like(g)

    for _ in range(iterations):
        grad_sub = cubic_gradient(g, H, w, 1e-3, M)
        w -= learning_rate*grad_sub

    return w




def cubic_gradient(g, H, w, alpha, M):

    grad = g + H @ w + (M/2)*np.linalg.norm(w)*w

    return grad
