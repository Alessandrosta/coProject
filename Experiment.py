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


def average_optimization(repetition, input):
    all_weights = []
    all_accuracy_lists = []
    all_loss_lists = []
    all_time_lists = []

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

    return avg_weights, avg_accuracy_list, avg_loss_list, avg_time_list


# Main function
def main():
    X_train, X_test, y_train, y_test = loadData.load_a9a()
    learning_rate = 0.5
    epochs = 50
    epsilon = 1e-4
    minibatch_size = 128

    # Store the functions in a dictionary
    functions1 = {
        'optimization_method': methods.stochastic_newton,
        'loss_function': mathFunctions.binary_cross_entropy_loss,
        'hessian_function': mathFunctions.logistic_loss_hessian,
        'gradient_function': mathFunctions.logistic_loss_gradient
    }

    # Average optimization results
    repetitions = 5  # Example: Run 5 repetitions
    # Define input parameters
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functions1
    }
    weights1, accuracy_list1, loss_list1, time_list1 = average_optimization(repetitions, input)


    # You can add more runs with different methods or parameters
    functions2 = {
        'optimization_method': methods.stochastic_cubic_newton,
        'loss_function': mathFunctions.binary_cross_entropy_loss,
        'hessian_function': mathFunctions.logistic_loss_hessian,
        'gradient_function': mathFunctions.logistic_loss_gradient
    }
    input = {
        'X': X_train,
        'y': y_train,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'minibatch_size': minibatch_size,
        'functions': functions2
    }
    weights2, accuracy_list2, loss_list2, time_list2 = average_optimization(repetitions, input)

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
        }
    ]

    # Generate plots
    makePlots.plot_loss_vs_iterations(runs, epsilon, filename='loss_diff_vs_iterations.png')
    makePlots.plot_loss_vs_time(runs, epsilon, filename='loss_diff_vs_time.png')
    makePlots.plot_accuracy_vs_iterations(runs, filename='accuracy_vs_iterations.png')
    makePlots.plot_accuracy_vs_time(runs, filename='accuracy_vs_time.png')
    


if __name__ == "__main__":
    main()
