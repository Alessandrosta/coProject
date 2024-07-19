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
    grad = logistic_loss_gradient(w, X, y, lambd, alpha)+grad_r

    return grad

def reg_loss_hessian(w, X, y, lambd, alpha, delta=1e-3):

    diag = lambd*(2*alpha - 6 * alpha * w ** 2)/((1 + alpha * w ** 2) ** 3)
    reg_hessian = logistic_loss_hessian(w, X, y,  lambd, alpha) + np.diag(diag)
    
    return reg_hessian


# Hv - from code

def logistic_loss_v2(w, X, Y, lambd, alpha, delta=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w) 
    l= - (np.dot(log_phi(z),Y)+np.dot(np.ones(n)-Y,one_minus_log_phi(z)))/n
    l = l + 0.5*  delta * (np.linalg.norm(w) ** 2)
    return l

def logistic_loss_nonconvex(w,X,Y,lambd, alpha, delta=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  # prediction <w, x>
    h = phi(z)
    l= - (np.dot(np.log(h),Y)+np.dot(np.ones(n)-Y,np.log(np.ones(n)-h)))/n
    l= l + lambd*np.dot(alpha*w**2,1/(1+alpha*w**2))
    return l

def logistic_loss_gradient_v2(w, X, Y, lambd, alpha, delta=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + delta * w
    return grad

def logist_loss_nonconvex_gradient(w, X, Y, lambd, alpha, delta=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)   # prediction <w, x>
    h = phi(z)
    grad= X.T.dot(h-Y)/n
    grad = grad + lambd*np.multiply(2*alpha*w,(1+alpha*w**2)**(-2))
    return grad

def logistic_loss_hessian_v2( w, X, Y, lambd, alpha, delta=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=phi(z)
    h= np.array(q*(1-phi(z)))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n
    H = H + delta * np.eye(d, d) 
    return H 

def logistic_loss_nonconvex_hessian( w, X, Y, lambd, alpha, delta=1e-3):
    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=phi(z)
    h= q*(1-phi(z))
    H = np.dot(np.transpose(X),h[:, np.newaxis]* X) / n  
    H = H + lambd * np.eye(d,d)*np.multiply(2*alpha-6*alpha**2*w**2,(alpha*w**2+1)**(-3))
    return H


def logistic_loss_Hv(w,X, Y, v,lambd, alpha, delta=1e-3): 
    n = X.shape[0]
    d = X.shape[1]
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + delta * v
    return out

def logistic_loss_nonconvex_Hv(w, X, Y, v, lambd, alpha): 
    n = X.shape[0]
    d = X.shape[1]
    _z=X.dot(w)
    _z = phi(-_z)
    d_binary = _z * (1 - _z)
    wa = d_binary * X.dot(v)
    Hv = X.T.dot(wa)/n
    out = Hv + lambd *np.multiply(np.multiply(2*alpha-6*alpha**2*w**2,(alpha*w**2+1)**(-3)), v)
    return out

# Sigmoid function
def softmax(z):
    z_exp = np.exp(z)
    softmax_probs = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    return softmax_probs

def sigmoid(z):
    return 1/(1+np.exp(-z))


######## Auxiliary Functions: robust Sigmoid, log(sigmoid) and 1-log(sigmoid) computations ########

def phi(t): #Author: Fabian Pedregosa
    # logistic function returns 1 / (1 + exp(-t))
    idx = t > 0
    out = np.empty(t.size, dtype=np.float64)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

def log_phi(t):
    # log(Sigmoid): log(1 / (1 + exp(-t)))
    idx = t>0
    out = np.empty(t.size, dtype=np.float64)
    out[idx]=-np.log(1+np.exp(-t[idx]))
    out[~idx]= t[~idx]-np.log(1+np.exp(t[~idx]))
    return out

def one_minus_log_phi(t):
    # log(1-Sigmoid): log(1-1 / (1 + exp(-t)))
    idx = t>0
    out = np.empty(t.size, dtype=np.float64)
    out[idx]= -t[idx]-np.log(1+np.exp(-t[idx]))
    out[~idx]=-np.log(1+np.exp(t[~idx]))
    return out
