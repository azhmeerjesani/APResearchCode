import numpy as np
import pandas as pd
import time
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, ReadoutError, QuantumError
from qiskit.providers.aer.noise.errors import (
    depolarizing_error,
    thermal_relaxation_error,
    coherent_unitary_error
)
# For real device noise model
# from qiskit_ibm_runtime import QiskitRuntimeService  # uncomment if you use IBM backends

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Sklearn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

###############################################################################
#                               CONFIG CONSTANTS
###############################################################################
N_EXPERIMENTS = 2  # Number of experiment repetitions

# Depolarizing noise rates
SINGLE_Q_DEPOL_ERROR = 0.01  # 1%
TWO_Q_DEPOL_ERROR = 0.03     # 3%

# Coherent noise offsets (approx. over-rotation)
Z_ROTATION_ERROR = np.pi / 30.0   # ~0.105 rad
X_ROTATION_ERROR = np.pi / 10.0   # ~0.314 rad

# Thermal Relaxation (Amplitude + Phase Damping)
T1_TIME = 50e-6  # 50 microseconds
T2_TIME = 30e-6  # 30 microseconds

# Gate durations (for amplitude/phase damping)
SINGLE_Q_GATE_TIME = 50e-9   # 50 ns
TWO_Q_GATE_TIME = 200e-9     # 200 ns

# Measurement Error
MEAS_ERROR_PROB = 0.02  # 2%

# Crosstalk noise (approx. correlated depolarizing)
CROSSTALK_DEPOL_ERROR = 0.02  # 2%

CSV_FILENAME = "qknn_iris_results.csv"  # Where results are stored

###############################################################################
#                            LOAD & PREPROCESS DATA
###############################################################################
def load_and_preprocess_iris():
    """
    Loads the Iris dataset and returns normalized features and labels.
    Normalization is L2 per-sample.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norms
    return X_norm, y

###############################################################################
#                      BUILDING A CUSTOM NOISE MODEL (Refactored)
###############################################################################
def build_custom_noise_model(
    depolarizing=True,
    coherent=True,
    thermal=True,
    measurement=True,
    crosstalk=True
):
    """
    Build and return a NoiseModel using the official docstring patterns:

    - Depolarizing noise via add_all_qubit_quantum_error()
    - Coherent over-rotation noise appended via custom quantum errors
    - Thermal relaxation error appended to single-qubit or two-qubit gates
    - Measurement error added to all qubits
    - Crosstalk approximated as correlated depolarizing on 2-qubit gates
    """

    noise_model = NoiseModel()

    # --- Depolarizing Errors ---
    if depolarizing:
        depol_error_1q = depolarizing_error(SINGLE_Q_DEPOL_ERROR, 1)
        depol_error_2q = depolarizing_error(TWO_Q_DEPOL_ERROR, 2)
        # Single-qubit gates: add to instructions 'x', 'y', 'z', etc.
        # For demonstration, let's assume a smaller gate set (common subset)
        single_qubit_gates = ["sx", "x", "rz", "h", "ry"]
        two_qubit_gates = ["cx", "cz", "swap", "cry", "crx", "rzz"]
        noise_model.add_all_qubit_quantum_error(depol_error_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(depol_error_2q, two_qubit_gates)

    # --- Coherent (Over-Rotation) Errors ---
    if coherent:
        # Over-rotation for single-qubit (append small Z rotation)
        # We'll define a quantum error that performs an extra Rz(Z_ROTATION_ERROR)
        qc_1q = QuantumCircuit(1)
        qc_1q.rz(Z_ROTATION_ERROR, 0)
        coherent_1q_err = QuantumError(coherent_unitary_error(qc_1q))

        # Over-rotation for two-qubit gates: an extra Rx(X_ROTATION_ERROR) on qubit 0
        qc_2q = QuantumCircuit(2)
        qc_2q.rx(X_ROTATION_ERROR, 0)
        coherent_2q_err = QuantumError(coherent_unitary_error(qc_2q))

        single_qubit_gates = ["sx", "x", "rz", "h", "ry"]
        two_qubit_gates = ["cx", "cz", "swap", "cry", "crx", "rzz"]
        noise_model.add_all_qubit_quantum_error(coherent_1q_err, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(coherent_2q_err, two_qubit_gates)

    # --- Thermal Relaxation (Amplitude + Phase Damping) ---
    if thermal:
        # Single-qubit thermal
        thermal_1q = thermal_relaxation_error(
            t1=T1_TIME,
            t2=T2_TIME,
            time=SINGLE_Q_GATE_TIME,
            excited_state_population=0
        )
        # Two-qubit thermal
        thermal_2q = thermal_relaxation_error(
            t1=T1_TIME,
            t2=T2_TIME,
            time=TWO_Q_GATE_TIME,
            excited_state_population=0
        )
        # We add these errors to each gate instruction as well:
        single_qubit_gates = ["sx", "x", "rz", "h", "ry"]
        two_qubit_gates = ["cx", "cz", "swap", "cry", "crx", "rzz"]
        noise_model.add_all_qubit_quantum_error(thermal_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(thermal_2q, two_qubit_gates)

    # --- Measurement Error ---
    if measurement:
        # Construct readout confusion matrix:
        #     0->0: 1 - p, 0->1: p
        #     1->0: p,     1->1: 1 - p
        meas_err = [[1 - MEAS_ERROR_PROB, MEAS_ERROR_PROB],
                    [MEAS_ERROR_PROB, 1 - MEAS_ERROR_PROB]]
        readout_err = ReadoutError(meas_err)
        noise_model.add_all_qubit_readout_error(readout_err)

    # --- Crosstalk Noise (Correlated Depolarizing on 2-qubit gates) ---
    if crosstalk:
        crosstalk_error = depolarizing_error(CROSSTALK_DEPOL_ERROR, 2)
        # Add to all two-qubit gates
        two_qubit_gates = ["cx", "cz", "swap", "cry", "crx", "rzz"]
        noise_model.add_all_qubit_quantum_error(crosstalk_error, two_qubit_gates)

    return noise_model

###############################################################################
#                           QUANTUM k-NN CLASSIFIER
###############################################################################
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    """
    Distance-based classification using a fidelity quantum kernel:
    distance ~ 1 - kernel_value
    """
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        # Distances
        distances = 1.0 - test_kernel_matrix[i, :]
        # Indices of k nearest neighbors
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        # Majority vote
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)

###############################################################################
#                      SINGLE EXPERIMENT RUN (LOOPS OVER k)
###############################################################################
def run_qknn_experiment(
    run_index,
    noise_label,
    noise_model,
    X_train, X_test,
    y_train, y_test,
    feature_map
):
    """
    Perform one run of the QkNN experiment, looping k from 1..len(X_test).
    """
    results_list = []
    start_time = time.time()

    # Create a simulator with or without noise
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    # Transpile feature map for target simulator
    transpiled_map = transpile(feature_map, simulator)
    # FidelityQuantumKernel with the transpiled map
    quantum_kernel = FidelityQuantumKernel(feature_map=transpiled_map, quantum_instance=simulator)

    # Compute kernel matrices
    train_kernel_start = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(X_train)
    train_kernel_end = time.time()

    test_kernel_start = time.time()
    test_kernel_matrix = quantum_kernel.evaluate(X_test, Y=X_train)
    test_kernel_end = time.time()

    # Evaluate QkNN for all k in [1 .. len(X_test)]
    test_size = X_test.shape[0]
    for k in range(1, test_size + 1):
        qknn_start = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k)
        qknn_end = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_time

        run_result = {
            "Run Number": run_index,
            "Noise Type": noise_label,
            "k": k,
            "ClassificationAccuracy": accuracy,
            "TrainKernelTime": train_kernel_end - train_kernel_start,
            "TestKernelTime": test_kernel_end - test_kernel_start,
            "QkNNTime": qknn_end - qknn_start,
            "TotalTimeSoFar": elapsed,
            "Date": datetime.now().strftime("%m/%d/%Y"),
            "Time": datetime.now().strftime("%H:%M:%S"),
        }
        results_list.append(run_result)

    return results_list

###############################################################################
#                            MAIN PIPELINE
###############################################################################
def main():
    """
    Main function that:
      1) Loads data and splits into train/test
      2) Builds feature map
      3) Iterates over different noise scenarios
      4) Logs results to CSV
    """

    # 1. Load data
    X, y = load_and_preprocess_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_features = X_train.shape[1]
    # Example feature map
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")

    # 2. Noise scenarios
    #   (label, booleans for custom noise toggles)
    noise_scenarios = [
        ("No Noise", (False, False, False, False, False)),
        ("Depolarizing Only", (True, False, False, False, False)),
        ("Coherent Only", (False, True, False, False, False)),
        ("Thermal Only", (False, False, True, False, False)),
        ("Measurement Only", (False, False, False, True, False)),
        ("Crosstalk Only", (False, False, False, False, True)),
        ("All Combined", (True, True, True, True, True))
    ]

    # 3. CSV header check
    write_header = False
    try:
        with open(CSV_FILENAME, 'r') as f:
            content = f.read().strip()
            if len(content) == 0:
                write_header = True
    except FileNotFoundError:
        write_header = True

    # 4. Run the experiments
    with open(CSV_FILENAME, 'a', newline='') as f:
        columns = [
            "Run Number", "Noise Type", "k",
            "ClassificationAccuracy", "TrainKernelTime",
            "TestKernelTime", "QkNNTime", "TotalTimeSoFar",
            "Date", "Time"
        ]
        if write_header:
            f.write(",".join(columns) + "\n")

        for scenario_label, scenario_flags in noise_scenarios:
            dep, coh, therm, meas, cross = scenario_flags

            for run_idx in range(1, N_EXPERIMENTS + 1):
                noise_model = None
                if any(scenario_flags):
                    noise_model = build_custom_noise_model(
                        depolarizing=dep,
                        coherent=coh,
                        thermal=therm,
                        measurement=meas,
                        crosstalk=cross
                    )

                # Run QkNN
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
                # Log results
                for r in results:
                    row_str = ",".join(str(r[col]) for col in columns)
                    f.write(row_str + "\n")
                f.flush()
                print(f"Completed: [{scenario_label}] Run {run_idx} (Test size = {len
