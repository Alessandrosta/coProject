import os
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_vs_iterations(runs, epsilon, filename='loss_vs_iterations.png', save_dir='Plots'):
    plt.figure()
    optimal_loss = min([min(run['loss_list']) for run in runs])

    for run in runs:
        iterations = np.arange(len(run['loss_list']))
        loss_diff = np.array(run['loss_list']) - optimal_loss + epsilon
        plt.plot(iterations, loss_diff, label=run['name'])

    plt.xlabel('Iterations')
    plt.ylabel('Loss - Optimal Loss')
    plt.yscale('log')
    plt.title('Loss Difference vs Iterations')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

def plot_loss_vs_time(runs, epsilon, filename='loss_vs_time.png', save_dir='Plots'):
    plt.figure()
    optimal_loss = min([min(run['loss_list']) for run in runs])

    for run in runs:
        time = run['time_list']
        loss_diff = np.array(run['loss_list']) - optimal_loss + epsilon
        plt.plot(time, loss_diff, label=run['name'])

    plt.xlabel('Time (s)')
    plt.ylabel('Loss - Optimal Loss')
    plt.yscale('log')
    plt.title('Loss Difference vs Time')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_vs_iterations(runs, filename='accuracy_vs_iterations.png', save_dir='Plots'):
    plt.figure()

    for run in runs:
        iterations = np.arange(len(run['accuracy_list']))
        plt.plot(iterations, run['accuracy_list'], label=run['name'])

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_vs_time(runs, filename='accuracy_vs_time.png', save_dir='Plots'):
    plt.figure()

    for run in runs:
        time = run['time_list']
        plt.plot(time, run['accuracy_list'], label=run['name'])

    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Time')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
