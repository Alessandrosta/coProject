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
    weights = np.ones((n, 1))*0.001
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
        w = np.ones(input['X'].shape[1])*0.001
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
def compare_Methods(dataset_name, loss, repetitions, learning_rate, epochs, minibatch_size, M, lambd, alpha, opt):
    
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
    
    
    # Define input parameters
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
            'name': 'Sub-sampled Cubic Regularization',
            'accuracy_list': accuracy_list3,
            'loss_list': loss_list3,
            'time_list': time_list3
        }
    ]
    
    return runs




def compare_Losses(dataset_name, repetitions, learning_rate, epochs, minibatch_size, M, lambd, alpha, opt):
    
    if dataset_name == 'a9a':
        X_train, X_test, y_train, y_test = loadData.load_a9a()
    elif dataset_name == 'ijcnn1':
        X_train, X_test, y_train, y_test = loadData.load_ijcnn1()
    else:
        print('Invalid Dataset choice! Choose between: a9a, ijcnn1 .')
        

    functionsSNL = {
        'optimization_method': methods.stochastic_newton,
        'loss_function': mathFunctions.binary_cross_entropy_loss,
        'hessian_function': mathFunctions.logistic_loss_hessian,
        'gradient_function': mathFunctions.logistic_loss_gradient
    }
    functionsSCNL = {
        'optimization_method': methods.stochastic_cubic_newton,
        'loss_function': mathFunctions.binary_cross_entropy_loss,
        'hessian_function': mathFunctions.logistic_loss_hessian,
        'gradient_function': mathFunctions.logistic_loss_gradient
    }
    functionsSCRL = {
        
        'loss_function': mathFunctions.logistic_loss_v2,
        'hessian_function': mathFunctions.logistic_loss_hessian_v2,
        'gradient_function': mathFunctions.logistic_loss_gradient_v2,
        'Hv': mathFunctions.logistic_loss_Hv
    }


    functionsSNLR = {
        'optimization_method': methods.stochastic_newton,
        'loss_function': mathFunctions.reg_loss,
        'hessian_function': mathFunctions.reg_loss_hessian,
        'gradient_function': mathFunctions.reg_loss_gradient
    }
    functionsSCNLR = {
        'optimization_method': methods.stochastic_cubic_newton,
        'loss_function': mathFunctions.reg_loss,
        'hessian_function': mathFunctions.reg_loss_hessian,
        'gradient_function': mathFunctions.reg_loss_gradient
        }
    functionsSCRLR = {
        'loss_function': mathFunctions.logistic_loss_nonconvex,
        'hessian_function': mathFunctions.logistic_loss_nonconvex_hessian,
        'gradient_function': mathFunctions.logist_loss_nonconvex_gradient,
        'Hv': mathFunctions.logistic_loss_nonconvex_Hv
    }
    

    # Define input parameters
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functionsSNL,
        'lambd': lambd,
        'alpha': alpha
    }
    weightsSNL, accuracy_listSNL, loss_listSNL, time_listSNL = average_optimization(repetitions, input, SCR=False)

    # You can add more runs with different methods or parameters
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functionsSNLR,
        'lambd': lambd,
        'alpha': alpha
    }
    weightsSNLR, accuracy_listSNLR, loss_listSNLR, time_listSNLR = average_optimization(repetitions, input, SCR=False)

    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functionsSCNL,
        'M': M,
        'lambd': lambd,
        'alpha': alpha
    }
    weightsSCNL, accuracy_listSCNL, loss_listSCNL, time_listSCNL = average_optimization(repetitions, input, SCR=False) 
    
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functionsSCNLR,
        'M': M,
        'lambd': lambd,
        'alpha': alpha
    }
    weightsSCNLR, accuracy_listSCNLR, loss_listSCNLR, time_listSCNLR = average_optimization(repetitions, input, SCR=False) 

    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functionsSCRL,
        'args': opt,
        'lambd': lambd,
        'alpha': alpha

    }
    weightsSCRL, accuracy_listSCRL, loss_listSCRL, time_listSCRL = average_optimization(repetitions, input, SCR=True) 

    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functionsSCRLR,
        'args': opt,
        'lambd': lambd,
        'alpha': alpha

    }
    weightsSCRLR, accuracy_listSCRLR, loss_listSCRLR, time_listSCRLR = average_optimization(repetitions, input, SCR=True) 


    # Store results for plotting
    runs = [
        {
            'name': 'SN Logistic',
            'accuracy_list': accuracy_listSNL,
            'loss_list': loss_listSNL,
            'time_list': time_listSNL
        },
        {
            'name': 'SN Regularized',
            'accuracy_list': accuracy_listSNLR,
            'loss_list': loss_listSNLR,
            'time_list': time_listSNLR
        },
        {
            'name': 'SCN Logistic',
            'accuracy_list': accuracy_listSCNL,
            'loss_list': loss_listSCNL,
            'time_list': time_listSCNL
        },
        {
            'name': 'SCN Regularized',
            'accuracy_list': accuracy_listSCNLR,
            'loss_list': loss_listSCNLR,
            'time_list': time_listSCNLR
        },
        {
            'name': 'SCR Logistic',
            'accuracy_list': accuracy_listSCRL,
            'loss_list': loss_listSCRL,
            'time_list': time_listSCRL
        },
        {
            'name': 'SCR Regularized',
            'accuracy_list': accuracy_listSCRLR,
            'loss_list': loss_listSCRLR,
            'time_list': time_listSCRLR
        }
    ]
    
    return runs


def compare_M_Values(dataset_name, loss, repetitions, learning_rate, epochs, minibatch_size, M, lambd, alpha, opt):
    
    if dataset_name == 'a9a':
        X_train, X_test, y_train, y_test = loadData.load_a9a()
    elif dataset_name == 'ijcnn1':
        X_train, X_test, y_train, y_test = loadData.load_ijcnn1()
    else:
        print('Invalid Dataset choice! Choose between: a9a, ijcnn1 .')
        
    if loss == 'Logistic loss':
        functions = {
            'optimization_method': methods.stochastic_cubic_newton,
            'loss_function': mathFunctions.binary_cross_entropy_loss,
            'hessian_function': mathFunctions.logistic_loss_hessian,
            'gradient_function': mathFunctions.logistic_loss_gradient
        }

    elif loss == 'Regularised logistic loss':
        functions = {
            'optimization_method': methods.stochastic_cubic_newton,
            'loss_function': mathFunctions.reg_loss,
            'hessian_function': mathFunctions.reg_loss_hessian,
            'gradient_function': mathFunctions.reg_loss_gradient
        }

    else:
        print('Invalid loss function choice! Choose between: Logistic loss, Regularised logistic loss .')
    
    
    runs = []
    # Store the functions in a dictionary
    for i in range(len(M)):
        Mtemp = M[i]
        input = {
            'X': X_train,
            'y': y_train,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'minibatch_size': minibatch_size,
            'functions': functions,
            'M': Mtemp,
            'lambd': lambd,
            'alpha': alpha
        }
        weights, accuracy_list, loss_list, time_list = average_optimization(repetitions, input, SCR=False)
        runs.append(
            {
            'name': f'SCN $M={Mtemp}$', 
            'accuracy_list': accuracy_list,
            'loss_list': loss_list,
            'time_list': time_list
        }
        )
    
    return runs

def compare_learningRates(dataset_name, loss, repetitions, learning_rate, epochs, minibatch_size, M, lambd, alpha, opt):
    
    if dataset_name == 'a9a':
        X_train, X_test, y_train, y_test = loadData.load_a9a()
    elif dataset_name == 'ijcnn1':
        X_train, X_test, y_train, y_test = loadData.load_ijcnn1()
    else:
        print('Invalid Dataset choice! Choose between: a9a, ijcnn1 .')
        
    if loss == 'Logistic loss':
        functions = {
            'optimization_method': methods.stochastic_newton,
            'loss_function': mathFunctions.binary_cross_entropy_loss,
            'hessian_function': mathFunctions.logistic_loss_hessian,
            'gradient_function': mathFunctions.logistic_loss_gradient
        }

    elif loss == 'Regularized logistic loss':
        functions = {
            'optimization_method': methods.stochastic_newton,
            'loss_function': mathFunctions.reg_loss,
            'hessian_function': mathFunctions.reg_loss_hessian,
            'gradient_function': mathFunctions.reg_loss_gradient
        }

    else:
        print('Invalid loss function choice! Choose between: Logistic loss, Regularised logistic loss .')
    
    
    runs = []
    # Store the functions in a dictionary
    for i in range(len(learning_rate)):
        l_temp = learning_rate[i]
        input = {
            'X': X_train,
            'y': y_train,
            'learning_rate': l_temp,
            'epochs': epochs,
            'minibatch_size': minibatch_size,
            'functions': functions,
            'M': M,
            'lambd': lambd,
            'alpha': alpha
        }
        weights, accuracy_list, loss_list, time_list = average_optimization(repetitions, input, SCR=False)
        runs.append(
            {
            'name': f'SN with $\lambda = {l_temp}$',
            'accuracy_list': accuracy_list,
            'loss_list': loss_list,
            'time_list': time_list
        }
        )
    
    return runs
#if __name__ == "__main__":
#    main()
