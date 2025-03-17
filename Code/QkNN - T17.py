# =============================================================================
# QUANTUM k-NN EXPERIMENTATION SCRIPT WITH MULTIPLE NOISE MODELS ON SEVERAL DATASETS
# =============================================================================
# This script demonstrates a quantum k-NN algorithm using Qiskit's FidelityQuantumKernel
# on multiple datasets (Iris, Wine, Breast Cancer, and Digits). It simulates various
# quantum noise environments by applying different noise models to the AerSimulator backend.
#
# The following noise models are supported:
#   1. "Noiseless"            : Ideal simulation with no noise.
#   2. "Noisy"                : Single-qubit depolarizing error (1% per gate).
#   3. "TwoQubit"             : Two-qubit depolarizing error (3% per gate on CX).
#   4. "ZRotation"            : Single-qubit coherent Z-rotation error (π/30 rotation).
#   5. "TwoQubitXRotation"    : Two-qubit coherent X-rotation error (π/10 rotation).
#   6. "T1Relaxation"         : Thermal relaxation noise with T1 randomly between 50-100 µs.
#   7. "T2Dephasing"          : Thermal relaxation noise simulating dephasing with T2 between 30-80 µs (with T1 very high).
#   8. "MeasurementError"     : Single-qubit measurement error with probability between 1%-3%.
#   9. "Combined"             : A combination of all the above noise models.
#
# The script:
#   - Loads and normalizes the dataset.
#   - Splits the data into training and testing sets.
#   - Constructs a ZZFeatureMap based on the number of features.
#   - Applies the selected noise model to an AerSimulator backend.
#   - Computes quantum kernel matrices for training and testing data.
#   - Runs a quantum k-NN classifier for a range of k values.
#   - Measures classification accuracy and records detailed timing information.
#   - Saves the results for each run to a CSV file with the dataset and noise type indicated.
# =============================================================================

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator  # For simulator with and without noise
from qiskit_aer.noise import (NoiseModel, depolarizing_error, coherent_unitary_error,
                              thermal_relaxation_error, ReadoutError)
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from scipy.stats import mode
import time
from datetime import datetime

# =============================================================================
# Set the number of repetitions for each experiment.
# =============================================================================
n = 1  # Change this variable to modify the number of runs

# =============================================================================
# Load datasets and prepare features and labels as pandas DataFrames/Series.
# =============================================================================

# Iris Dataset
iris = load_iris()
iris_features = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_labels = pd.Series(iris.target, name="label")

# Wine Dataset
wine = load_wine()
wine_features = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_labels = pd.Series(wine.target, name="label")

# Breast Cancer Dataset
breast_cancer = load_breast_cancer()
breast_cancer_features = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
breast_cancer_labels = pd.Series(breast_cancer.target, name="label")

# Digits Dataset (no feature names available)
digits = load_digits()
digits_features = pd.DataFrame(digits.data)
digits_labels = pd.Series(digits.target, name="label")

# Create a dictionary mapping dataset names to their features and labels.
datasets = {
    "Iris": (iris_features, iris_labels),
    "Wine": (wine_features, wine_labels),
    "BreastCancer": (breast_cancer_features, breast_cancer_labels),
    "Digits": (digits_features, digits_labels)
}


# =============================================================================
# FUNCTION: quantum_knn
# Description:
#   Implements the quantum k-NN classification algorithm. For each test sample,
#   it computes the distance (1 - kernel value) to all training samples,
#   selects the k nearest neighbors, and determines the most common label.
#
# Parameters:
#   - test_kernel_matrix: Precomputed kernel matrix for test samples.
#   - train_kernel_matrix: Precomputed kernel matrix for training samples.
#   - y_train: Training labels (as a pandas Series).
#   - k: Number of nearest neighbors to consider (default is 3).
#
# Returns:
#   - A NumPy array of predicted labels for the test samples.
# =============================================================================
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1 - test_kernel_matrix[i, :]
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train.iloc[k_nearest_indices].values
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)


# =============================================================================
# FUNCTION: run_qknn_experiment
# Description:
#   Runs a single experiment of the quantum k-NN algorithm on a given dataset,
#   applying a specified noise model. It performs the following steps:
#
#     1. Data normalization.
#     2. Splitting data into training and testing sets.
#     3. Constructing a quantum feature map (ZZFeatureMap).
#     4. Setting up the AerSimulator backend with the chosen noise model.
#     5. Computing the quantum kernel matrices for training and testing data.
#     6. Running the quantum k-NN classification for varying values of k.
#     7. Recording timing and accuracy metrics.
#
# Parameters:
#   - run_index: Identifier for the current run.
#   - noise_type: String specifying the noise model to use.
#   - features: Pandas DataFrame of features.
#   - labels: Pandas Series of labels.
#
# Returns:
#   - results_list: A list of dictionaries containing timing, accuracy, noise type,
#                   and other info.
# =============================================================================
def run_qknn_experiment(run_index, noise_type="Noiseless", features=None, labels=None):
    if features is None or labels is None:
        raise ValueError("features and labels must be provided")

    # Record start time and normalize data
    start_time = time.time()
    norm_start_time = time.time()
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    norm_end_time = time.time()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_features, labels, test_size=0.2
    )

    # Define Quantum Feature Map (ZZFeatureMap with linear entanglement)
    num_features = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")

    # =============================================================================
    # Noise Model Selection:
    #   Depending on the 'noise_type' parameter, a specific noise model is applied to
    #   the AerSimulator backend.
    # =============================================================================
    if noise_type == "Noisy":
        noise_model = NoiseModel()
        error = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "TwoQubit":
        noise_model = NoiseModel()
        error = depolarizing_error(0.03, 2)
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "ZRotation":
        theta = np.pi / 30
        U = np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]])
        error = coherent_unitary_error(U)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "TwoQubitXRotation":
        theta = np.pi / 10
        Rx = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                       [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
        U = np.kron(Rx, Rx)
        error = coherent_unitary_error(U)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "T1Relaxation":
        noise_model = NoiseModel()
        t1 = np.random.uniform(50000, 100000)  # T1 in ns (50-100 µs)
        t2 = t1  # For simplicity, T2 is set equal to T1
        gate_time_1q = 50  # Typical single-qubit gate time (ns)
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
        gate_time_2q = 300  # Typical two-qubit gate time (ns)
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q)
        error_2q = error_2q.tensor(error_2q)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "T2Dephasing":
        noise_model = NoiseModel()
        t2 = np.random.uniform(30000, 80000)  # T2 in ns (30-80 µs)
        t1 = 1e9  # T1 is set very high to minimize amplitude damping
        gate_time_1q = 50
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3"])
        gate_time_2q = 300
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q)
        error_2q = error_2q.tensor(error_2q)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "MeasurementError":
        noise_model = NoiseModel()
        error_prob = np.random.uniform(0.01, 0.03)
        readout_error = ReadoutError([[1 - error_prob, error_prob],
                                      [error_prob, 1 - error_prob]])
        noise_model.add_all_qubit_readout_error(readout_error)
        backend = AerSimulator(noise_model=noise_model)
    elif noise_type == "Combined":
        noise_model = NoiseModel()
        # Single-qubit depolarizing error (1%)
        error_1q_dep = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error_1q_dep, ["u1", "u2", "u3"])
        # Two-qubit depolarizing error (3%)
        error_2q_dep = depolarizing_error(0.03, 2)
        noise_model.add_all_qubit_quantum_error(error_2q_dep, ["cx"])
        # Single-qubit Z-rotation error (π/30)
        theta_z = np.pi / 30
        U_z = np.array([[np.exp(-1j * theta_z / 2), 0],
                        [0, np.exp(1j * theta_z / 2)]])
        error_z = coherent_unitary_error(U_z)
        noise_model.add_all_qubit_quantum_error(error_z, ["u1", "u2", "u3"])
        # Two-qubit X-rotation error (π/10)
        theta_x = np.pi / 10
        Rx = np.array([[np.cos(theta_x / 2), -1j * np.sin(theta_x / 2)],
                       [-1j * np.sin(theta_x / 2), np.cos(theta_x / 2)]])
        U_x = np.kron(Rx, Rx)
        error_x = coherent_unitary_error(U_x)
        noise_model.add_all_qubit_quantum_error(error_x, ["cx"])
        # T1 relaxation noise (T1 between 50-100 µs, T2 = T1)
        t1 = np.random.uniform(50000, 100000)
        t2_t1 = t1
        gate_time_1q = 50
        error_1q_t1 = thermal_relaxation_error(t1, t2_t1, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q_t1, ["u1", "u2", "u3"])
        gate_time_2q = 300
        error_2q_t1 = thermal_relaxation_error(t1, t2_t1, gate_time_2q)
        error_2q_t1 = error_2q_t1.tensor(error_2q_t1)
        noise_model.add_all_qubit_quantum_error(error_2q_t1, ["cx"])
        # T2 dephasing noise (T2 between 30-80 µs, T1 very high)
        t2_dephase = np.random.uniform(30000, 80000)
        t1_dephase = 1e9
        error_1q_t2 = thermal_relaxation_error(t1_dephase, t2_dephase, gate_time_1q)
        noise_model.add_all_qubit_quantum_error(error_1q_t2, ["u1", "u2", "u3"])
        error_2q_t2 = thermal_relaxation_error(t1_dephase, t2_dephase, gate_time_2q)
        error_2q_t2 = error_2q_t2.tensor(error_2q_t2)
        noise_model.add_all_qubit_quantum_error(error_2q_t2, ["cx"])
        # Measurement error (error probability between 1%-3%)
        error_prob_meas = np.random.uniform(0.01, 0.03)
        readout_error = ReadoutError([[1 - error_prob_meas, error_prob_meas],
                                      [error_prob_meas, 1 - error_prob_meas]])
        noise_model.add_all_qubit_readout_error(readout_error)
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = AerSimulator()

    # =============================================================================
    # Kernel Matrix Computation:
    #   The FidelityQuantumKernel is used to compute the quantum kernel matrices for
    #   both the training data and the testing data.
    # =============================================================================
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    train_kernel_start_time = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
    train_kernel_end_time = time.time()

    test_kernel_start_time = time.time()
    test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    test_kernel_end_time = time.time()

    # =============================================================================
    # Quantum k-NN Classification:
    #   For each value of k from 1 up to the number of test samples, the quantum k-NN
    #   classifier is applied to predict the test labels. The accuracy is then computed.
    # =============================================================================
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


# =============================================================================
# EXPERIMENT EXECUTION:
#   For each dataset (Iris, Wine, BreastCancer, and Digits), the experiment is run
#   for each noise condition. For each condition, the experiment is run 'n' times,
#   and the results are saved in CSV files with filenames indicating the dataset and noise type.
# =============================================================================

# List of noise types to iterate over
noise_types = ["Noiseless", "Noisy", "TwoQubit", "ZRotation", "TwoQubitXRotation",
               "T1Relaxation", "T2Dephasing", "MeasurementError", "Combined"]

for dataset_name, (ds_features, ds_labels) in datasets.items():
    for noise in noise_types:
        for i in range(1, n + 1):
            results = run_qknn_experiment(i, noise_type=noise, features=ds_features, labels=ds_labels)
            results_df = pd.DataFrame(results)
            output_file = f"{dataset_name}_QkNN_Run_{i}_{noise}_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Run {i} ({noise}) on {dataset_name} dataset completed. Results saved to {output_file}.")
