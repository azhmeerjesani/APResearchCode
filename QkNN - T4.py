import numpy as np
import pandas as pd
import time
from datetime import datetime

# -----------------------------
# Qiskit and Qiskit-Aer imports
# -----------------------------
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap

# -----------------------------
# IBM Quantum Runtime (optional)
# for retrieving real backend noise
# -----------------------------
# If you have an IBM Quantum account/token, you can set:
#    pip install qiskit_ibm_runtime
#    from qiskit_ibm_runtime import QiskitRuntimeService
#    service = QiskitRuntimeService(channel='ibm_quantum', token='REPLACE_WITH_YOUR_TOKEN')
#    backend = service.get_backend('ibm_kyoto')  # or any other backend name
#
# Then:
#    noise_model = NoiseModel.from_backend(backend)
#
# For demonstration below, we will show how to set "noise_model = None" or
# use the device noise model if you have an actual backend.
# -----------------------------

# -----------------------------
# Scikit-learn imports
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from scipy.stats import mode

###############################################################################
#                       CONFIGURATION & GLOBAL CONSTANTS
###############################################################################

# Output CSV
CSV_FILENAME = "qknn_iris_results.csv"

# Number of experiment repetitions
N_EXPERIMENTS = 1  # Adjust as needed

###############################################################################
#                        DATA LOADING & PREPROCESSING
###############################################################################

def load_and_preprocess_iris():
    """
    Loads the Iris dataset and returns normalized features (L2 norm) and labels.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # L2 normalization
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norm

    return X_norm, y


###############################################################################
#                           QkNN CLASSIFIER
###############################################################################

def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    """
    Given the precomputed fidelity-based kernel matrices, classify test samples
    using a standard k-NN majority vote approach with distance ~ (1 - Fidelity).
    """
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        # "Distance" = 1 - fidelity
        distances = 1.0 - test_kernel_matrix[i, :]
        # Find the indices of the k nearest neighbors
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        # Majority vote among nearest neighbors
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)

###############################################################################
#                          EXPERIMENT FUNCTION
###############################################################################

def run_qknn_experiment(
    run_index,
    noise_label,
    noise_model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_map
):
    """
    Runs a single QkNN experiment on the given train/test split with the specified
    noise model. Loops k from 1..len(X_test) to gather accuracy metrics for each k.
    Returns a list of result dictionaries (one for each k).
    """
    results_list = []
    start_time = time.time()

    # Create or configure the simulator with/without noise model
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    # Transpile feature map for the simulator
    transpiled_fmap = transpile(feature_map, simulator, optimization_level=0)

    # Fidelity-based quantum kernel
    quantum_kernel = FidelityQuantumKernel(
        feature_map=transpiled_fmap,
        quantum_instance=simulator
    )

    # Compute Kernel Matrices
    train_kernel_start = time.time()
    train_matrix = quantum_kernel.evaluate(X_train)
    train_kernel_end = time.time()

    test_kernel_start = time.time()
    test_matrix = quantum_kernel.evaluate(X_test, Y=X_train)
    test_kernel_end = time.time()

    # For each k in [1..test_size], compute QkNN accuracy
    test_size = X_test.shape[0]
    for k in range(1, test_size + 1):
        qknn_start = time.time()
        y_pred = quantum_knn(test_matrix, train_matrix, y_train, k=k)
        qknn_end = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_time

        result = {
            "Run Number"           : run_index,
            "Noise Type"           : noise_label,
            "k"                    : k,
            "ClassificationAccuracy": accuracy,
            "TrainKernelTime"      : train_kernel_end - train_kernel_start,
            "TestKernelTime"       : test_kernel_end - test_kernel_start,
            "QkNNTime"             : qknn_end - qknn_start,
            "TotalTimeSoFar"       : elapsed,
            "Date"                 : datetime.now().strftime("%m/%d/%Y"),
            "Time"                 : datetime.now().strftime("%H:%M:%S"),
        }
        results_list.append(result)

    return results_list

###############################################################################
#                             MAIN PIPELINE
###############################################################################

def main():
    """
    Main pipeline to compare:
      - Baseline QkNN (no noise)
      - QkNN with real device noise from an IBM backend (if available)
    Logs each run (and each k) to a CSV file.
    """

    # 1. Load data
    X, y = load_and_preprocess_iris()

    # 2. Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature map
    num_features = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')

    # 4. Optional: Retrieve real device noise model
    # Uncomment lines below if you have an IBM Quantum account
    # -----------------------------------------------------------------
    # from qiskit_ibm_runtime import QiskitRuntimeService
    # service = QiskitRuntimeService(channel='ibm_quantum', token='REPLACE_WITH_YOUR_TOKEN')
    # backend = service.get_backend('ibm_kyoto')  # or any available backend
    # real_device_noise_model = NoiseModel.from_backend(backend)
    # -----------------------------------------------------------------
    # For demonstration, we'll set it to None here. Replace with real noise model in practice:
    real_device_noise_model = None  # e.g. = NoiseModel.from_backend(backend)

    # 5. Scenarios
    #    a) "No Noise"
    #    b) "IBM Device Noise"
    scenarios = [
        ("No Noise", None),
        ("Real IBM Noise", real_device_noise_model),
    ]

    # Check if we need to create CSV file headers
    write_header = False
    try:
        with open(CSV_FILENAME, 'r') as f:
            if len(f.read().strip()) == 0:
                write_header = True
    except FileNotFoundError:
        write_header = True

    # 6. Run each scenario for N_EXPERIMENTS
    with open(CSV_FILENAME, 'a', newline='') as f:
        columns = [
            "Run Number", "Noise Type", "k",
            "ClassificationAccuracy", "TrainKernelTime",
            "TestKernelTime", "QkNNTime", "TotalTimeSoFar",
            "Date", "Time"
        ]
        if write_header:
            f.write(",".join(columns) + "\n")

        for scenario_label, noise_model in scenarios:
            for run_idx in range(1, N_EXPERIMENTS + 1):
                results = run_qknn_experiment(
                    run_index=run_idx,
                    noise_label=scenario_label,
                    noise_model=noise_model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_map=feature_map
                )
                # Write results to CSV
                for r in results:
                    row_str = ",".join(str(r[c]) for c in columns)
                    f.write(row_str + "\n")
                f.flush()

                print(f"[{scenario_label}] Run {run_idx} completed. Logged to {CSV_FILENAME}.")

    print("\nAll experiments finished. Check CSV for results.\n")


if __name__ == "__main__":
    main()
