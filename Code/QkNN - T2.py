import numpy as np
import pandas as pd
import time
from datetime import datetime

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer  # Correct import for Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import (depolarizing_error,
                                               thermal_relaxation_error,
                                               pauli_error,
                                               coherent_unitary_error)
from qiskit.providers.aer.noise.errors.quantum_error import QuantumError
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Sklearn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

###############################################################################
#                          CONFIGURATION VARIABLES
###############################################################################

# Number of experiment repetitions
N_EXPERIMENTS = 2  # change as needed

# You can modify these default noise parameters as needed or expose them as CLI args.

# Depolarizing noise rates (Pauli errors)
SINGLE_Q_DEPOL_ERROR = 0.01  # 1%
TWO_Q_DEPOL_ERROR = 0.03  # 3%

# Coherent noise (systematic over-rotation)
# Single-qubit Z rotation offset; two-qubit X rotation offset (radians)
Z_ROTATION_ERROR = np.pi / 30.0  # ~0.105 rad
X_ROTATION_ERROR = np.pi / 10.0  # ~0.314 rad

# Amplitude + Phase damping via Thermal Relaxation
# T1, T2 in microseconds (convert to seconds for Qiskit)
T1_TIME = 50e-6
T2_TIME = 30e-6

# Approximate gate times for single & two-qubit gates (seconds)
SINGLE_Q_GATE_TIME = 50e-9  # 50 ns
TWO_Q_GATE_TIME = 200e-9  # 200 ns

# Measurement Error
# Probability of measuring |0> as |1> or |1> as |0>
MEAS_ERROR_PROB = 0.02  # 2%

# Crosstalk noise: approximate as correlated depolarizing between pairs of qubits
CROSSTALK_DEPOL_ERROR = 0.02  # 2% (example)

# CSV filename
CSV_FILENAME = "qknn_iris_results.csv"


###############################################################################
#                           DATA LOADING & PREPROCESSING
###############################################################################

def load_and_preprocess_iris():
    """
    Loads the Iris dataset and returns normalized features (L2 norm) and labels.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target

    # L2 normalization of each sample
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / norm

    return X_norm, y


###############################################################################
#                              NOISE MODEL CREATION
###############################################################################

def create_noise_model(
        use_depolarizing=False,
        use_coherent=False,
        use_thermal_relaxation=False,
        use_measurement_error=False,
        use_crosstalk=False
):
    """
    Creates and returns a Qiskit NoiseModel() with the specified noise components.
    Each noise source can be toggled via boolean flags.
    Default noise levels are taken from global constants at top of the script.
    """
    noise_model = NoiseModel()

    # ------------------- Depolarizing Noise -------------------
    # Single-qubit
    if use_depolarizing:
        depol_error_1q = depolarizing_error(SINGLE_Q_DEPOL_ERROR, 1)
        depol_error_2q = depolarizing_error(TWO_Q_DEPOL_ERROR, 2)
        # Add errors to all single-qubit gates
        for gate in ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u"]:
            noise_model.add_quantum_error(depol_error_1q, gate, [0])
            noise_model.add_quantum_error(depol_error_1q, gate, [1])
        # Add errors to all two-qubit gates
        for gate in ["cx", "cz", "swap", "cry", "crx", "rzz"]:
            noise_model.add_quantum_error(depol_error_2q, gate, [0, 1])

    # ------------------- Coherent Noise (Over-Rotation) -------------------
    # We'll approximate by adding a small Z or X rotation after each gate.
    # Single-qubit: add a small Z rotation error
    # Two-qubit: add a small X rotation error
    if use_coherent:
        # Over-rotation for single-qubit gates
        z_overrotation = QuantumError(coherent_unitary_error(
            QuantumCircuit(1).rz(Z_ROTATION_ERROR, 0).to_gate()
        ))
        # Over-rotation for two-qubit gates
        x_overrotation_2q = QuantumError(coherent_unitary_error(
            QuantumCircuit(2).rx(X_ROTATION_ERROR, 0).to_gate()
        ))
        for gate in ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u"]:
            noise_model.add_quantum_error(z_overrotation, gate, [0])
            noise_model.add_quantum_error(z_overrotation, gate, [1])
        for gate in ["cx", "cz", "swap", "cry", "crx", "rzz"]:
            noise_model.add_quantum_error(x_overrotation_2q, gate, [0, 1])

    # ------------------- Thermal Relaxation (Amplitude+Phase Damping) -------------------
    if use_thermal_relaxation:
        # Single-qubit gate thermal relaxation
        single_qubit_thermal = thermal_relaxation_error(
            t1=T1_TIME,
            t2=T2_TIME,
            time=SINGLE_Q_GATE_TIME,
            excited_state_population=0
        )
        # Two-qubit gate thermal relaxation
        two_qubit_thermal = thermal_relaxation_error(
            t1=T1_TIME,
            t2=T2_TIME,
            time=TWO_Q_GATE_TIME,
            excited_state_population=0
        )

        for gate in ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "u"]:
            noise_model.add_quantum_error(single_qubit_thermal, gate, [0])
            noise_model.add_quantum_error(single_qubit_thermal, gate, [1])
        for gate in ["cx", "cz", "swap", "cry", "crx", "rzz"]:
            noise_model.add_quantum_error(two_qubit_thermal, gate, [0, 1])

    # ------------------- Measurement Error -------------------
    if use_measurement_error:
        # Probability of flipping measurement outcomes
        # Construct a 2x2 matrix:
        # [[1 - p, p],
        #  [p,     1 - p]]
        meas_error_prob = MEAS_ERROR_PROB
        prob0to1 = meas_error_prob
        prob1to0 = meas_error_prob
        # Qiskit wants a list of lists: [ [p(0|0), p(0|1)], [p(1|0), p(1|1)] ]
        # i.e. confusion matrix rows are "reported outcome", columns are "real outcome"
        measurement_confusion = [[1 - prob0to1, prob0to1],
                                 [prob1to0, 1 - prob1to0]]
        noise_model.add_readout_error(measurement_confusion, [0])
        noise_model.add_readout_error(measurement_confusion, [1])

    # ------------------- Crosstalk Noise (Approx. Correlated Depolarizing) -------------------
    # For demonstration, we add an extra correlated depolarizing on two-qubit gates
    if use_crosstalk:
        correlated_depol = depolarizing_error(CROSSTALK_DEPOL_ERROR, 2)
        for gate in ["cx", "cz", "swap", "cry", "crx", "rzz"]:
            noise_model.add_quantum_error(correlated_depol, gate, [0, 1])

    return noise_model


###############################################################################
#                               QkNN CLASSIFIER
###############################################################################

def quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=3):
    """
    Given the precomputed kernel matrix between test and train sets (test_kernel_matrix)
    and the train-train kernel matrix (train_kernel_matrix), classify via a k-NN approach.
    """
    predictions = []
    for i in range(test_kernel_matrix.shape[0]):
        # The "distance" in fidelity-based kernel can be approximated by (1 - K).
        distances = 1.0 - test_kernel_matrix[i, :]
        k_nearest_indices = distances.argsort()[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        # Use majority vote among the k nearest labels
        predicted_label = mode(k_nearest_labels, keepdims=True).mode[0]
        predictions.append(predicted_label)
    return np.array(predictions)


###############################################################################
#                              EXPERIMENT FUNCTION
###############################################################################

def run_qknn_experiment(
        run_index,
        noise_label,
        noise_model=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        feature_map=None,
        backend_options=None
):
    """
    Runs a single QkNN experiment on the given train/test split with the specified noise model.
    Loops k from 1..len(X_test) to gather accuracy metrics for each k.
    Returns a list of result dictionaries (one for each k).
    """
    results_list = []

    start_time = time.time()

    # Set up quantum kernel with the provided noise model or None
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()  # no noise

    # Transpile feature_map for the simulator backend if needed
    transpiled_fmap = transpile(feature_map, simulator)

    quantum_kernel = FidelityQuantumKernel(feature_map=transpiled_fmap, quantum_instance=simulator)

    # Compute kernel matrices
    train_kernel_start = time.time()
    train_kernel_matrix = quantum_kernel.evaluate(X_train)
    train_kernel_end = time.time()

    test_kernel_start = time.time()
    test_kernel_matrix = quantum_kernel.evaluate(X_test, Y=X_train)
    test_kernel_end = time.time()

    # For each k in [1..test_size], compute QkNN accuracy
    test_size = X_test.shape[0]
    for k in range(1, test_size + 1):
        qknn_start = time.time()
        y_pred = quantum_knn(test_kernel_matrix, train_kernel_matrix, y_train, k=k)
        qknn_end = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_time

        # Prepare result record
        result = {
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
        results_list.append(result)

    return results_list


###############################################################################
#                           MAIN EXPERIMENT PIPELINE
###############################################################################

def main():
    """
    Main pipeline to run:
      - Baseline (no noise)
      - Individual noise sources
      - All noise combined
    For each, run N_EXPERIMENTS times and log results incrementally to CSV.
    """
    # ------------------ 1. Load Data, Train/Test Split ------------------
    X, y = load_and_preprocess_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_features = X_train.shape[1]

    # Create a feature map for QkNN
    # Example: 2 repetitions, linear entanglement
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')

    # ------------------ 2. Prepare noise scenarios ------------------
    # (label, bools for each noise type) so we can loop easily
    noise_scenarios = [
        ("No Noise", (False, False, False, False, False)),
        ("Depolarizing Only", (True, False, False, False, False)),
        ("Coherent Only", (False, True, False, False, False)),
        ("ThermalRelaxation Only", (False, False, True, False, False)),
        ("MeasurementError Only", (False, False, False, True, False)),
        ("Crosstalk Only", (False, False, False, False, True)),
        ("All Noise Combined", (True, True, True, True, True))
    ]

    # If CSV file doesn't exist or is empty, write header
    # Otherwise, we will append results
    write_header = False
    try:
        with open(CSV_FILENAME, 'r') as f:
            # check if it's empty
            if len(f.read().strip()) == 0:
                write_header = True
    except FileNotFoundError:
        write_header = True

    with open(CSV_FILENAME, 'a', newline='') as f:
        # We’ll use pandas for convenience to handle row-by-row writing
        # but you can also use csv.writer if you prefer
        columns = [
            "Run Number", "Noise Type", "k",
            "ClassificationAccuracy", "TrainKernelTime",
            "TestKernelTime", "QkNNTime", "TotalTimeSoFar",
            "Date", "Time"
        ]
        if write_header:
            f.write(",".join(columns) + "\n")

        # ------------------ 3. Run each scenario for N_EXPERIMENTS -------------
        for scenario_label, scenario_flags in noise_scenarios:
            dep_flag, coh_flag, therm_flag, meas_flag, cross_flag = scenario_flags

            for run_idx in range(1, N_EXPERIMENTS + 1):
                # Create noise model
                noise_model = None
                if any(scenario_flags):
                    noise_model = create_noise_model(
                        use_depolarizing=dep_flag,
                        use_coherent=coh_flag,
                        use_thermal_relaxation=therm_flag,
                        use_measurement_error=meas_flag,
                        use_crosstalk=cross_flag
                    )

                # Run QkNN experiment
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

                # Write each result row into CSV
                for r in results:
                    row_str = ",".join(str(r[c]) for c in columns)
                    f.write(row_str + "\n")
                f.flush()  # flush to ensure data is written after each run

                print(f"Finished: [{scenario_label}] Run {run_idx} with test size {len(X_test)}")

    print(f"\nAll experiments completed. Results are logged in '{CSV_FILENAME}'.\n")


###############################################################################
#                              ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
