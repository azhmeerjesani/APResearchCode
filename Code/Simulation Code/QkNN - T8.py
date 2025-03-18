import numpy as np
import pandas as pd
from qiskit_aer import Aer  # Correct import for Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from scipy.stats import mode
import time
from datetime import datetime

# Define the number of repetitions
n = 5  # Change this variable to modify the number of runs

# Load the Iris dataset
iris = load_iris()
features = pd.DataFrame(iris.data, columns=iris.feature_names)
labels = pd.Series(iris.target, name="label")


# Quantum k-NN function
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1 - test_kernel_matrix[i, :]
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train.iloc[k_nearest_indices].values
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)


# Function to perform a single run and return the results for all k values
def run_qknn_experiment(run_index):
    results = {}

    # Record start time
    start_time = time.time()

    # Normalize the data
    norm_start_time = time.time()
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    norm_end_time = time.time()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2, random_state=42
    )

    # Define Quantum Feature Map
    num_features = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")

    # Define Quantum Kernel without specifying quantum_instance
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Compute Kernel Matrices
    train_kernel_start_time = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
    train_kernel_end_time = time.time()

    test_kernel_start_time = time.time()
    test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    test_kernel_end_time = time.time()

    # Create a list to hold results for each k value
    results_list = []

    # Loop through all k values from 1 to the test set size
    for k in range(1, X_test.shape[0] + 1):
        qknn_start_time = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k)
        qknn_end_time = time.time()

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        result = {
            'Run Index': run_index,
            'k': k,
            'Normalization Time': norm_end_time - norm_start_time,
            'Train Kernel Time': train_kernel_end_time - train_kernel_start_time,
            'Test Kernel Time': test_kernel_end_time - test_kernel_start_time,
            'QkNN Time': qknn_end_time - qknn_start_time,
            'Accuracy': accuracy,
            'Date': datetime.now().strftime("%m/%d/%Y"),
            'Time': datetime.now().strftime("%H:%M:%S")
        }
        results_list.append(result)

    return results_list


# Repeat the experiment `n` times and save results to CSV files
for i in range(1, n + 1):
    results = run_qknn_experiment(i)
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} completed. Results saved to {output_file}.")
