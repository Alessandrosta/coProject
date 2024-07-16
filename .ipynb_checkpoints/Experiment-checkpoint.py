import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import random
import time
import os
import methods
import mathFunctions
import loadData
import makePlots
import scr






def optimization(X, y, learning_rate, epochs, minibatch_size, functions, M=0.02, lambd=0.001, alpha=1):
    # Access the functions
    optimization_method = functions['optimization_method']
    loss_function = functions['loss_function']
    hessian_function = functions['hessian_function']
    gradient_function = functions['gradient_function']

    m, n = X.shape
    weights = np.ones((n, 1))*0.1
    #bias = np.zeros((1, 1))

    #initialize lists
    accuracy_list = []
    loss_list = []
    time_list =[]
    # Compute accuracy
    accuracy = compute_accuracy(X, y, weights)
    accuracy_list.append(accuracy)
    print(f'Accuracy epoch 0: {accuracy}')

    loss_list.append(loss_function(weights, X, y, lambd, alpha))
    time_list.append(0)
    # Start measuring time
    start_time = time.time()
    for epoch in range(epochs):
        # Linear combination
        #linear_output = np.dot(X, weights)
        # Activation function
        #predictions = sigmoid(linear_output)

        # Compute loss with given loss function
        #loss = Loss_function(weights, X, y, lambd, alpha)
        
        # Do a optimization step with given method
        weights, loss = optimization_method(X, y, weights, learning_rate, minibatch_size,
                                             hessian_function , gradient_function, loss_function , M, lambd, alpha)
        
        # save loss
        loss_list.append(loss)
        # Compute accuracy
        accuracy = compute_accuracy(X, y, weights)
        accuracy_list.append(accuracy)
        #compute time
        iter_time = time.time() - start_time
        time_list.append(iter_time)


        # print time for the first 10 epochs
        if epoch == 9:  # epoch is zero-indexed, so this checks after 10 iterations
            # Estimate time for all epochs
            expected_total_time = iter_time / 10 * epochs
            # Print out the measured time and expected total time
            print(f"Time for the first 10 epochs: {iter_time:.4f} seconds")
            print(f"Expected total time for {epochs} epochs: {expected_total_time:.2f} seconds")

        # Print accuracy 10 times
        if epoch == 0 or (epoch + 1) % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

    end_time = time.time() - start_time
    print(f'total time: {end_time:.2f} seconds')    
    return weights, accuracy_list, loss_list, time_list

# Accuracy computation
def compute_accuracy(X, y, weights):
    linear_output = np.dot(X, weights)
    predictions = mathFunctions.sigmoid(linear_output)
    predicted_labels = np.zeros(predictions.shape[0])
    for i in range(predictions.shape[0]):
        if predictions[i] > 0.5:
            predicted_labels[i] = 1
    accuracy = np.mean(predicted_labels == y)
    return accuracy


def average_optimization(repetition, input, SCR):
    all_weights = []
    all_accuracy_lists = []
    all_loss_lists = []
    all_time_lists = []
    
    if SCR == False:
        for _ in range(repetition):
            # Run optimization
            weights, accuracy_list, loss_list, time_list = optimization(**input)

            # Store results for averaging
            all_weights.append(weights)
            all_accuracy_lists.append(accuracy_list)
            all_loss_lists.append(loss_list)
            all_time_lists.append(time_list)

        # Compute averages
        avg_weights = np.mean(all_weights, axis=0)
        avg_accuracy_list = np.mean(all_accuracy_lists, axis=0)
        avg_loss_list = np.mean(all_loss_lists, axis=0)
        avg_time_list = np.mean(all_time_lists, axis=0)
    
    elif SCR == True:
        w = np.zeros(input['X'].shape[1])
        func = input['functions']
        loss = func['loss_function']
        gradient = func['gradient_function']
        hessian = func['hessian_function']
        opt = input['args']
        X = input['X']
        Y = input['y']
        Hv = func['Hv']
        
        for _ in range(repetition):
            weights, accuracy_list, loss_list, time_list, sample_list = scr.SCR(w, loss, gradient, Hv, hessian, X, Y, opt)
            # Store results for averaging
            all_weights.append(weights)
            all_accuracy_lists.append(accuracy_list)
            all_loss_lists.append(loss_list)
            all_time_lists.append(time_list)

            # Compute averages
        avg_weights = np.mean(all_weights, axis=0)
        avg_accuracy_list = np.mean(all_accuracy_lists, axis=0)
        avg_loss_list = np.mean(all_loss_lists, axis=0)
        avg_time_list = np.mean(all_time_lists, axis=0)

    return avg_weights, avg_accuracy_list, avg_loss_list, avg_time_list


# Main function
def execution(dataset_name, loss, repetitions, learning_rate, epochs, epsilon, minibatch_size, M, lambd, alpha, opt):
    
    
    if dataset_name == 'a9a':
        X_train, X_test, y_train, y_test = loadData.load_a9a()
    elif dataset_name == 'ijcnn1':
        X_train, X_test, y_train, y_test = loadData.load_ijcnn1()
    else:
        print('Invalid Dataset choice! Choose between: a9a, ijcnn1 .')
        
    
    
    
    if loss == 'Logistic loss':
        functions1 = {
            'optimization_method': methods.stochastic_newton,
            'loss_function': mathFunctions.binary_cross_entropy_loss,
            'hessian_function': mathFunctions.logistic_loss_hessian,
            'gradient_function': mathFunctions.logistic_loss_gradient
        }
        functions2 = {
            'optimization_method': methods.stochastic_cubic_newton,
            'loss_function': mathFunctions.binary_cross_entropy_loss,
            'hessian_function': mathFunctions.logistic_loss_hessian,
            'gradient_function': mathFunctions.logistic_loss_gradient
        }
        functions3 = {
            
            'loss_function': mathFunctions.logistic_loss_v2,
            'hessian_function': mathFunctions.logistic_loss_hessian_v2,
            'gradient_function': mathFunctions.logistic_loss_gradient_v2,
            'Hv': mathFunctions.logistic_loss_Hv
        }

    elif loss == 'Regularised logistic loss':
        functions1 = {
            'optimization_method': methods.stochastic_newton,
            'loss_function': mathFunctions.reg_loss,
            'hessian_function': mathFunctions.reg_loss_hessian,
            'gradient_function': mathFunctions.reg_loss_gradient
        }
        functions2 = {
            'optimization_method': methods.stochastic_cubic_newton,
            'loss_function': mathFunctions.reg_loss,
            'hessian_function': mathFunctions.reg_loss_hessian,
            'gradient_function': mathFunctions.reg_loss_gradient
        }
        functions3 = {
            
            'loss_function': mathFunctions.logistic_loss_nonconvex,
            'hessian_function': mathFunctions.logistic_loss_nonconvex_hessian,
            'gradient_function': mathFunctions.logist_loss_nonconvex_gradient,
            'Hv': mathFunctions.logistic_loss_nonconvex_Hv
        }

    else:
        print('Invalid loss function choice! Choose between: Logistic loss, Regularised logistic loss .')
    
    
    learning_rate = learning_rate
    epochs = epochs
    epsilon = epsilon
    minibatch_size = minibatch_size
    M = M
    lambd = lambd
    alpha = alpha

    # Store the functions in a dictionary

    # Average optimization results
    repetitions = repetitions  # Example: Run 5 repetitions
    # Define input parameters
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functions3,
        'args': opt,
        'lambd': lambd,
        'alpha': alpha

    }
    weights3, accuracy_list3, loss_list3, time_list3 = average_optimization(repetitions, input, SCR=True) 
    
    
    
    
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functions1,
        'lambd': lambd,
        'alpha': alpha
    }
    weights1, accuracy_list1, loss_list1, time_list1 = average_optimization(repetitions, input, SCR=False)


    # You can add more runs with different methods or parameters
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functions2,
        'M': M,
        'lambd': lambd,
        'alpha': alpha
    }
    weights2, accuracy_list2, loss_list2, time_list2 = average_optimization(repetitions, input, SCR=False)


    
    # Store results for plotting
    runs = [
        {
            'name': 'Stochastic Newton',
            'accuracy_list': accuracy_list1,
            'loss_list': loss_list1,
            'time_list': time_list1
        },
        {
            'name': 'Stochastic cubic Newton',
            'accuracy_list': accuracy_list2,
            'loss_list': loss_list2,
            'time_list': time_list2
        },
        {
            'name': 'Sub-sampled Regularised Cubic Newton',
            'accuracy_list': accuracy_list3,
            'loss_list': loss_list3,
            'time_list': time_list3
        }
    ]
    
    return runs


    


#if __name__ == "__main__":
#    main()
