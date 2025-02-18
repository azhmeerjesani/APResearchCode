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

# Load the IRIS dataset and set up features and labels
iris = load_iris()
features = pd.DataFrame(iris.data, columns=iris.feature_names)
labels = pd.Series(iris.target, name="label")

# Define the Quantum k-NN function that computes distances in kernel space and votes on the label
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        # Convert the kernel similarity into a distance metric
        distances = 1 - test_kernel_matrix[i, :]
        # Find the indices of the k nearest neighbors (lowest distances)
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train.iloc[k_nearest_indices].values
        # Majority vote for the predicted label
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)

# Function to perform one run of the QkNN experiment and return the results
def run_qknn_experiment(run_index):
    results = {}

    # Record start time
    start_time = time.time()

    # Normalize the data (each sample is normalized to unit length)
    norm_start_time = time.time()
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    norm_end_time = time.time()

    # Split dataset into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2, random_state=42
    )

    # Define the Quantum Feature Map using the ZZFeatureMap
    num_features = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")

    # Create the quantum kernel based on the fidelity between quantum states
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Compute the kernel matrix for the training data
    train_kernel_start_time = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
    train_kernel_end_time = time.time()

    # Compute the kernel matrix between the test data and the training data
    test_kernel_start_time = time.time()
    test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    test_kernel_end_time = time.time()

    # Run the Quantum k-NN classification (using k=5 here)
    qknn_start_time = time.time()
    y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=5)
    qknn_end_time = time.time()

    # Calculate classification accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Record timings and accuracy
    results['Run Index'] = run_index
    results['Normalization Time'] = norm_end_time - norm_start_time
    results['Train Kernel Time'] = train_kernel_end_time - train_kernel_start_time
    results['Test Kernel Time'] = test_kernel_end_time - test_kernel_start_time
    results['QkNN Time'] = qknn_end_time - qknn_start_time
    results['Accuracy'] = accuracy
    results['Date'] = datetime.now().strftime("%m/%d/%Y")
    results['Time'] = datetime.now().strftime("%H:%M:%S")

    return results

# Number of runs to perform
n = 5  # You can change this value to perform more or fewer runs

# Repeat the experiment n times and save the results to CSV files
for i in range(1, n + 1):
    results = run_qknn_experiment(i)
    results_df = pd.DataFrame([results])
    output_file = f"Iris_QkNN_Run_{i}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} completed. Results saved to {output_file}.")
