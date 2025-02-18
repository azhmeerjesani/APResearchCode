import numpy as np
import pandas as pd
from qiskit_aer import Aer, AerSimulator  # For simulator with and without noise
from qiskit_aer.noise import (NoiseModel, depolarizing_error, coherent_unitary_error,
                              thermal_relaxation_error, ReadoutError)
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
    # Record start time and normalize data
    start_time = time.time()
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
        # Single-qubit X-rotation: R_x(θ) = cos(θ/2) I - i sin(θ/2) X
        Rx = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                       [-1j * np.sin(theta/2), np.cos(theta/2)]])
        # Two-qubit error as tensor product of two Rx rotations
        U = np.kron(Rx, Rx)
        error = coherent_unitary_error(U)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "T1Relaxation":
        # T1 relaxation noise with T1 randomly chosen between 50-100 µs (in ns)
        noise_model = NoiseModel()
        t1 = np.random.uniform(50000, 100000)  # T1 in ns (50-100 µs)
        t2 = t1  # For simplicity, set T2 equal to T1
        gate_time_1q = 50    # Typical single-qubit gate time in ns
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
        gate_time_2q = 300   # Typical two-qubit gate time in ns
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q)
        error_2q = error_2q.tensor(error_2q)  # Construct two-qubit error from two single-qubit errors
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "T2Dephasing":
        # T2 dephasing noise with T2 randomly chosen between 30-80 µs (in ns)
        # To simulate dephasing only, set T1 very high.
        noise_model = NoiseModel()
        t2 = np.random.uniform(30000, 80000)  # T2 in ns (30-80 µs)
        t1 = 1e9  # Set T1 very high so that amplitude damping is negligible.
        gate_time_1q = 50    # Typical single-qubit gate time in ns
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
        gate_time_2q = 300   # Typical two-qubit gate time in ns
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q)
        error_2q = error_2q.tensor(error_2q)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "MeasurementError":
        # Single-qubit measurement error with error probability randomly chosen between 1%-3%
        noise_model = NoiseModel()
        error_prob = np.random.uniform(0.01, 0.03)
        readout_error = ReadoutError([[1 - error_prob, error_prob],
                                      [error_prob, 1 - error_prob]])
        noise_model.add_all_qubit_readout_error(readout_error)
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
    for k in range(1, X_test.shape[0] + 1):
        qknn_start_time = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k)
        qknn_end_time = time.time()
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


# Run experiments for all eight noise conditions

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

# 6. T1 relaxation runs (T1 relaxation time: 50-100 µs)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="T1Relaxation")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_T1Relaxation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (T1Relaxation) completed. Results saved to {output_file}.")

# 7. T2 dephasing runs (T2 dephasing time: 30-80 µs)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="T2Dephasing")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_T2Dephasing_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (T2Dephasing) completed. Results saved to {output_file}.")

# 8. Measurement error runs (single-qubit measurement error with probability 1%-3%)
for i in range(1, n + 1):
    results = run_qknn_experiment(i, noise_type="MeasurementError")
    results_df = pd.DataFrame(results)
    output_file = f"Iris_QkNN_Run_{i}_MeasurementError_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Run {i} (MeasurementError) completed. Results saved to {output_file}.")
