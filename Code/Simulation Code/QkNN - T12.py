import numpy as np
import pandas as pd
from qiskit_aer import Aer  # Correct import for Aer
from qiskit_aer import AerSimulator  # For simulator with noise
from qiskit_aer.noise import NoiseModel, depolarizing_error, coherent_unitary_error  # For noise models
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from scipy.stats import mode
import time
from datetime import datetime

# Define the number of repetitions
n = 1  # Change this variable to modify the number of runs

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


# Function to perform a single run and return the results for all k values.
# The `noise_type` parameter controls which noise model to apply.
def run_qknn_experiment(run_index, noise_type="Noiseless"):
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

    # Set up the simulator backend based on the noise type.
    if noise_type == "Noisy":
        # Single-qubit depolarizing error: 1% per gate
        noise_model = NoiseModel()
        error = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "TwoQubit":
        # Two-qubit depolarizing error: 3% per gate
        noise_model = NoiseModel()
        error = depolarizing_error(0.03, 2)
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "ZRotation":
        # Single-qubit coherent Z-rotation error of π/30 applied to all single-qubit gates
        theta = np.pi / 30
        U = np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]])
        error = coherent_unitary_error(U)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "TwoQubitXRotation":
        # Two-qubit coherent X-rotation error of π/10 applied to all two-qubit gates (cx)
        theta = np.pi / 10
        # Define single-qubit X-rotation matrix: R_x(θ) = cos(θ/2) I - i sin(θ/2) X
        Rx = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                       [-1j * np.sin(theta/2), np.cos(theta/2)]])
        # Create a two-qubit noise operator as the tensor product of two single-qubit rotations
        U = np.kron(Rx, Rx)
        error = coherent_unitary_error(U)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    else:
        # No noise is applied.
        backend = AerSimulator()

    # Define Quantum Kernel with the specified backend
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
            'Noise': noise_type,
            'Date': datetime.now().strftime("%m/%d/%Y"),
            'Time': datetime.now().strftime("%H:%M:%S")
        }
        results_list.append(result)

    return results_list


# Run experiments for all five noise conditions

# 1. Noiseless runs
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="Noiseless")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_Noiseless_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (Noiseless) completed. Results saved to {output_file}.")

# 2. Single-qubit error runs (1% depolarizing error per gate)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="Noisy")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_Noisy_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (Noisy - Single-qubit error) completed. Results saved to {output_file}.")

# 3. Two-qubit error runs (3% depolarizing error per gate)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="TwoQubit")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_TwoQubit_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (TwoQubit) completed. Results saved to {output_file}.")

# 4. Single-qubit Z-rotation noise runs (π/30 rotation error)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="ZRotation")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_ZRotation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (ZRotation) completed. Results saved to {output_file}.")

# 5. Two-qubit X-rotation noise runs (π/10 rotation error)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="TwoQubitXRotation")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_TwoQubitXRotation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (TwoQubitXRotation) completed. Results saved to {output_file}.")
