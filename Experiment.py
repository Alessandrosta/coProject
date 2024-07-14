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

    X_train, y_train = load_svmlight_file(current_file_directory+'\\a9a.txt')
    X_test, y_test = load_svmlight_file(current_file_directory+'\\a9a.t')
    #X_train, y_train = load_svmlight_file('a9a.txt')
    #X_test, y_test = load_svmlight_file('a9a.t')

    X_train = X_train.toarray()
    y_train= [0 if e == -1 else e for e in y_train]
    y_train=np.array(y_train)

    X_test = X_test.toarray()
    y_test= [0 if e == -1 else e for e in y_test]
    y_test=np.array(y_test)
    return X_train, X_test, y_train, y_test

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


def logistic_loss_gradient(w, X, Y, delta=1e-3):

    n = X.shape[0]
    d = X.shape[1]
    z = X.dot(w)  
    h = sigmoid(z)
    grad= X.T.dot(h-Y[:, np.newaxis])/n
    grad = grad + delta * w

    return  grad

def logistic_loss_hessian( w, X, Y, delta=1e-3):

    n = X.shape[0]
    d = X.shape[1]
    z= X.dot(w)
    q=sigmoid(z)
    h= np.array(q*(1-sigmoid(z)))
    H = np.dot(np.transpose(X),h* X) / n
    H = H + delta * np.eye(d, d)

    return H


# Regularized loss function
def regularized_loss(w, X, y, lambd, alpha):

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

#def gradient_descent(X, y, predictions, weights, bias, learning_rate, minibatch_size, hessian_function, gradient_function):
#    m,_ = X.shape
#    # Gradient computation
#    error = predictions.T - y
#    weight_gradient = np.dot(X.T, error.T) / m
#    bias_gradient = np.mean(error)
#    
#    # Update weights and bias
#    weights -= learning_rate * weight_gradient
#    bias -= learning_rate * bias_gradient
#    return weights, bias

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
    Hk = hessian_function(weights, X_batch, y_batch)
    gk = gradient_function(weights, X_batch, y_batch)

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
    Hk = hessian_function(weights, X_batch, y_batch)
    gk = gradient_function(weights, X_batch, y_batch)
    

    loss = 1000
    # update only if successful
    old_prediction = sigmoid(np.dot(X, weights))
    old_loss = Loss_function(weights, X, y, lambd, alpha)
    new_weights = weights + cubic_regularization_subproblem_gd(gk, Hk, M, learning_rate=0.1, iterations=100)
    new_prediction = sigmoid(np.dot(X, new_weights))
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







def optimization(X, y, optimization_method, Loss_function, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    m, n = X.shape
    weights = np.ones((n, 1))*0.1
    #bias = np.zeros((1, 1))
    accuracy_list = []
    loss_list = []
    # Compute accuracy
    accuracy = compute_accuracy(X, y, weights)
    accuracy_list.append(accuracy)
    print(f'Accuracy epoch 0: {accuracy}')
    for epoch in range(epochs):
        # Linear combination
        #linear_output = np.dot(X, weights)
        # Activation function
        #predictions = sigmoid(linear_output)

        # Compute loss with given loss function
        #loss = Loss_function(weights, X, y, lambd, alpha)
        
        # Do a optimization step with given method
        weights, loss = optimization_method(X, y, weights, learning_rate, minibatch_size = 1000,
                                             hessian_function = logistic_loss_hessian, gradient_function = logistic_loss_gradient, Loss_function = Loss_function, M=0.02, lambd=0.001, alpha=1)
        
        loss_list.append(loss)
        # Compute accuracy
        accuracy = compute_accuracy(X, y, weights)
        accuracy_list.append(accuracy)
        
        # Print accuracy 10 times
        if epoch == 0 or (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')
        
    return weights, accuracy_list, loss_list

# Accuracy computation
def compute_accuracy(X, y, weights):
    linear_output = np.dot(X, weights)
    predictions = sigmoid(linear_output)
    predicted_labels = np.zeros(predictions.shape[0])
    for i in range(predictions.shape[0]):
        if predictions[i] > 0.5:
            predicted_labels[i] = 1
    accuracy = np.mean(predicted_labels == y)
    return accuracy

# Main function
def main():
    X_train, X_test, y_train, y_test = load_a9a()
    learning_rate = 1
    epochs = 1000
    #choose from = stochastic_newton, stochastic_cubic_newton
    optimization_method = stochastic_newton


    print(f'size Xtrain: {X_train.shape}')
    print(f'size ytrain: {y_train.shape}')
    print(f'size Xtest: {X_test.shape}')
    print(f'size yTest: {y_test.shape}')
    print(f"Start Training with Learning Rate: {learning_rate} ...")
    weights, accuracy_list, loss_list = optimization(X_train, y_train, optimization_method, regularized_loss, learning_rate, epochs)


    #print("Starting Test...")
    # Test accuracy
    #test_accuracy = compute_accuracy(X_test, y_test, weights, bias)
    #print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Plotting accuracy vs iterations
    plt.plot(accuracy_list)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Iterations for {optimization_method.__name__}')
    plt.savefig('accuracy_plot.png')  # Save plot as PNG file
    plt.close()  # Close the plot window

    print("Finished Training")



    # Compute f^* (the best loss observed)
    f_star = min(loss_list)

    # Compute f(x^k) - f^* for each epoch
    fxk_minus_f_star = [loss - f_star for loss in loss_list]

    # Add a small epsilon value to avoid log(0)
    epsilon = 1e-5
    fxk_minus_f_star = [loss + epsilon for loss in fxk_minus_f_star]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), fxk_minus_f_star, label='$f(x^k) - f^* + \epsilon$')
    plt.yscale('log')  # Use a logarithmic scale for better visualization of small values
    plt.xlabel('Epochs')
    plt.ylabel('$f(x^k) - f^* + \epsilon$')
    plt.title(f'Convergence of {optimization_method.__name__}')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig('plotfk_fstar.png')

    # Close the plot to avoid displaying it in an interactive environment
    plt.close()
    


if __name__ == "__main__":
    main()
