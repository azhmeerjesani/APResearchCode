#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    coherent_unitary_error,
    thermal_relaxation_error,
    ReadoutError,
    QuantumError
)
from qiskit.circuit.library import ZZFeatureMap

# The new way: we create a custom fidelity object via the "ComputeUncompute" approach
# and a "Sampler" that references our AerSimulator.
# (Ensure qiskit_machine_learning >= 0.5 and qiskit >= 0.22 for these imports.)
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels.algorithms import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# scikit-learn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

###############################################################################
#                          CONFIGURATION VARIABLES
###############################################################################
N_EXPERIMENTS = 1
CSV_FILENAME = "qknn_iris_results.csv"

# Depolarizing error rates
SINGLE_Q_DEPOL_ERROR = 0.01
TWO_Q_DEPOL_ERROR    = 0.03

# Coherent noise
Z_ROTATION_ERROR = np.pi / 30.0
X_ROTATION_ERROR = np.pi / 10.0

# Thermal relaxation
T1_TIME = 50e-6
T2_TIME = 30e-6
SINGLE_Q_GATE_TIME = 50e-9
TWO_Q_GATE_TIME    = 200e-9

# Measurement error
MEAS_ERROR_PROB = 0.02

# Crosstalk noise
CROSSTALK_DEPOL_ERROR = 0.02

###############################################################################
#                          DATA LOADING & PREPROCESSING
###############################################################################
def load_and_preprocess_iris():
    print("[INFO] Loading the IRIS dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norms
    print("[INFO] Iris dataset loaded and normalized.")
    return X_norm, y

###############################################################################
#                           NOISE MODEL CREATION
###############################################################################
def create_noise_model(
    use_depolarizing=False,
    use_coherent=False,
    use_thermal_relaxation=False,
    use_measurement_error=False,
    use_crosstalk=False
):
    print("[INFO] Creating noise model...")
    noise_model = NoiseModel()

    single_qubit_gates = ["id", "rz", "sx", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "u"]
    two_qubit_gates    = ["cx", "cz", "swap", "crx", "cry", "rzz"]

    if use_depolarizing:
        print("  - Adding depolarizing noise")
        depol_1q = depolarizing_error(SINGLE_Q_DEPOL_ERROR, 1)
        depol_2q = depolarizing_error(TWO_Q_DEPOL_ERROR, 2)
        noise_model.add_all_qubit_quantum_error(depol_1q, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(depol_2q, two_qubit_gates)

    if use_coherent:
        print("  - Adding coherent (over-rotation) noise")
        z_err_circ = QuantumCircuit(1)
        z_err_circ.rz(Z_ROTATION_ERROR, 0)
        single_coherent_err = QuantumError(coherent_unitary_error(z_err_circ))

        x_err_circ_2q = QuantumCircuit(2)
        x_err_circ_2q.rx(X_ROTATION_ERROR, 0)
        twoq_coherent_err = QuantumError(coherent_unitary_error(x_err_circ_2q))

        noise_model.add_all_qubit_quantum_error(single_coherent_err, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(twoq_coherent_err, two_qubit_gates)

    if use_thermal_relaxation:
        print("  - Adding thermal relaxation noise")
        error_1q_thermal = thermal_relaxation_error(
            t1=T1_TIME, t2=T2_TIME,
            time=SINGLE_Q_GATE_TIME,
            excited_state_population=0.0
        )
        error_2q_thermal = thermal_relaxation_error(
            t1=T1_TIME, t2=T2_TIME,
            time=TWO_Q_GATE_TIME,
            excited_state_population=0.0
        )
        noise_model.add_all_qubit_quantum_error(error_1q_thermal, single_qubit_gates)
        noise_model.add_all_qubit_quantum_error(error_2q_thermal, two_qubit_gates)

    if use_measurement_error:
        print("  - Adding measurement error")
        p0to1 = MEAS_ERROR_PROB
        p1to0 = MEAS_ERROR_PROB
        confusion_matrix = [
            [1 - p0to1, p0to1],
            [p1to0,     1 - p1to0]
        ]
        meas_error = ReadoutError(confusion_matrix)
        noise_model.add_all_qubit_readout_error(meas_error)

    if use_crosstalk:
        print("  - Adding crosstalk noise (2-qubit correlated depolarizing)")
        crosstalk_error_2q = depolarizing_error(CROSSTALK_DEPOL_ERROR, 2)
        noise_model.add_all_qubit_quantum_error(crosstalk_error_2q, two_qubit_gates)

    print("[INFO] Noise model created.\n")
    return noise_model

###############################################################################
#                           QkNN HELPER
###############################################################################
def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        distances = 1.0 - test_kernel_matrix[i, :]
        k_indices = distances.argsort()[:k]
        k_labels  = y_train[k_indices]
        label     = mode(k_labels, keepdims=True).mode[0]
        predictions.append(label)
    return np.array(predictions)

###############################################################################
#                       QkNN EXPERIMENT FUNCTION
###############################################################################
def run_qknn_experiment(
    run_index,
    noise_label,
    noise_model=None,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    feature_map=None
):
    """
    We now build a FidelityQuantumKernel using a custom fidelity object that
    references our desired AerSimulator (with or without noise).
    """
    print(f"[INFO] Starting QkNN experiment for run {run_index} with noise='{noise_label}'...")

    # Create or use a no-noise simulator
    if noise_model:
        simulator = AerSimulator(noise_model=noise_model)
        print("  - AerSimulator created WITH noise.")
    else:
        simulator = AerSimulator()
        print("  - AerSimulator created WITHOUT noise.")

    # Create a Sampler that uses this simulator
    sampler = Sampler(backend=simulator)
    # Build a ComputeUncompute fidelity object
    fidelity = ComputeUncompute(sampler=sampler)

    # Transpile the feature map if desired
    print("  - Transpiling feature map...")
    transpiled_fmap = transpile(feature_map, simulator)

    # Build the FidelityQuantumKernel with our custom fidelity
    print("  - Creating FidelityQuantumKernel with custom fidelity...")
    quantum_kernel = FidelityQuantumKernel(
        feature_map=transpiled_fmap,
        fidelity=fidelity
    )

    print("  - Computing train kernel matrix...")
    start_time = time.time()
    tk_start = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(X_train)  # no backend arg
    tk_end = time.time()

    print("  - Computing test kernel matrix...")
    te_start = time.time()
    test_kernel_matrix = quantum_kernel.evaluate(X_test, y_vec=X_train)  # no backend arg
    te_end = time.time()

    results_list = []
    test_size = X_test.shape[0]
    print(f"  - Classifying with k in [1..{test_size}]...")

    for k_val in range(1, test_size + 1):
        qk_start = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k_val)
        qk_end = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        total_elapsed = time.time() - start_time

        results_list.append({
            "Run Number": run_index,
            "Noise Type": noise_label,
            "k": k_val,
            "ClassificationAccuracy": accuracy,
            "TrainKernelTime": tk_end - tk_start,
            "TestKernelTime": te_end - te_start,
            "QkNNTime": qk_end - qk_start,
            "TotalTimeSoFar": total_elapsed,
            "Date": datetime.now().strftime("%m/%d/%Y"),
            "Time": datetime.now().strftime("%H:%M:%S")
        })

    print(f"[INFO] Experiment finished for run {run_index}, noise='{noise_label}'.\n")
    return results_list

###############################################################################
#                              MAIN FUNCTION
###############################################################################
def main():
    print("[INFO] BEGINNING MAIN PIPELINE")

    # 1. Load data
    X, y = load_and_preprocess_iris()
    print("[INFO] Splitting dataset into train/test with 80/20 ratio.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Create feature map
    num_features = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement="linear")
    print("[INFO] ZZFeatureMap created with dimension=", num_features)

    # 3. Define noise scenarios
    noise_scenarios = [
        ("No Noise",                (False, False, False, False, False)),
        ("Depolarizing Only",       (True,  False, False, False, False)),
        ("Coherent Only",           (False, True,  False, False, False)),
        ("ThermalRelaxation Only",  (False, False, True,  False, False)),
        ("MeasurementError Only",   (False, False, False, True,  False)),
        ("Crosstalk Only",          (False, False, False, False, True)),
        ("All Noise Combined",      (True,  True,  True,  True,  True))
    ]

    # 4. Prepare CSV for logging
    print(f"[INFO] Checking if '{CSV_FILENAME}' exists or is empty. Will append results.")
    write_header = False
    try:
        with open(CSV_FILENAME, 'r') as f:
            if len(f.read().strip()) == 0:
                write_header = True
    except FileNotFoundError:
        write_header = True

    columns = [
        "Run Number", "Noise Type", "k",
        "ClassificationAccuracy",
        "TrainKernelTime", "TestKernelTime", "QkNNTime",
        "TotalTimeSoFar", "Date", "Time"
    ]

    print(f"[INFO] Starting experiments. Total scenarios: {len(noise_scenarios)}")
    with open(CSV_FILENAME, 'a', newline='') as csvfile:
        if write_header:
            csvfile.write(",".join(columns) + "\n")

        for scenario_label, flags in noise_scenarios:
            dep_flag, coh_flag, therm_flag, meas_flag, cross_flag = flags

            for run_idx in range(1, N_EXPERIMENTS + 1):
                print(f"[INFO] Building noise model for scenario: '{scenario_label}' (Run {run_idx})...")
                scenario_model = None
                if any(flags):
                    scenario_model = create_noise_model(
                        use_depolarizing=dep_flag,
                        use_coherent=coh_flag,
                        use_thermal_relaxation=therm_flag,
                        use_measurement_error=meas_flag,
                        use_crosstalk=cross_flag
                    )

                print("[INFO] Running QkNN experiment now...")
                results = run_qknn_experiment(
                    run_index=run_idx,
                    noise_label=scenario_label,
                    noise_model=scenario_model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_map=feature_map
                )

                for row in results:
                    row_str = ",".join(str(row[c]) for c in columns)
                    csvfile.write(row_str + "\n")
                csvfile.flush()

                print(f"[INFO] Completed scenario='{scenario_label}', run={run_idx}.\n")

    print(f"[INFO] All experiments finished. Results stored in '{CSV_FILENAME}'.\n")

###############################################################################
#                              SCRIPT ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
