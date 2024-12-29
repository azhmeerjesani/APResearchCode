################################################################################
# QkNN on IRIS dataset with Qiskit Aer
# ------------------------------------
# This is an illustrative notebook example showing how you might implement
# a simple QkNN approach using Qiskit for the distance measurement.
################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Qiskit
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error
from qiskit.providers.aer.noise.errors import ReadoutError

# For IRIS dataset and splitting
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

###############################################################################
# 1) Load IRIS dataset, 80/20 train/test split
###############################################################################
iris = load_iris()
X = iris.data  # shape (150, 4)
y = iris.target  # shape (150, )
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# For simplicity, we can do a small dimension reduction from 4D to 2D using the first 2 features
# OR you can attempt to amplitude-encode 4D in a 2-qubit or 3-qubit circuit.
# Below, we’ll just keep features 0 and 1 for demonstration:
X = X[:, :2]  # keep only 2 features for a simpler amplitude encoding

# Normalize data to [0,1] range
scaler = MinMaxScaler()
X = scaler.fit_transform(X)  # each feature in range [0,1]

# Create 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)

num_train = len(X_train)
num_test = len(X_test)
print(f"Training set size: {num_train}, Test set size: {num_test}")


###############################################################################
# 2) Define a function to build a quantum circuit that encodes a data point
#    in amplitude form.
#
#    We assume X has 2 features [f1, f2]. We map them to 2-qubit amplitude:
#        state = f1|0> + f2|1>, normalized
#
#    In practice, you need proper normalization and more sophisticated encoding
#    if your dataset has >2 features or if values exceed [0,1].
###############################################################################
def encode_point_amplitude(f1, f2):
    """
    Build a 1-qubit circuit that encodes a 2D data point
    into amplitudes of a single qubit (or 2-qubit if you prefer).

    For simplicity, we assume f1^2 + f2^2 <= 1
    If not, we will re-normalize them.
    """
    norm = np.sqrt(f1 ** 2 + f2 ** 2)
    if norm < 1e-9:
        f1n, f2n = 1.0, 0.0
    else:
        f1n, f2n = f1 / norm, f2 / norm

    qc = QuantumCircuit(1, name="encode")
    # Initialize qubit in amplitude |psi> = f1n|0> + f2n|1>
    # We can do this via qc.initialize([f1n, f2n], 0)
    qc.initialize([f1n, f2n], 0)

    return qc


###############################################################################
# 3) Define a function to measure "distance" or "similarity" between two points
#    using an overlap test (like a swap test, but simpler).
#
#    We'll do:
#       Encode point A in qubit0,
#       Encode point B in qubit1,
#       Then measure overlap by applying e.g. a partial swap test or
#       a simpler approach:
#           1) We create 2 qubits, encode A in qubit0, B in qubit1
#           2) Then we measure fidelity as Probability of measuring |00> or |11>
#              after a few gates that entangle them, etc.
#
#    For a truly rigorous distance measure, you'd do a SWAP test on an ancilla,
#    but below is a minimal example to get the idea across.
###############################################################################
def build_distance_circuit(f1A, f2A, f1B, f2B):
    """
    Return a circuit that attempts to measure overlap between state(A) & state(B).
    A smaller 'overlap' means a bigger distance, etc.
    """
    qc = QuantumCircuit(2)

    # Encode A on qubit0
    normA = np.sqrt(f1A ** 2 + f2A ** 2)
    if normA < 1e-9:
        f1nA, f2nA = 1.0, 0.0
    else:
        f1nA, f2nA = f1A / normA, f2A / normA
    qc.initialize([f1nA, f2nA], 0)

    # Encode B on qubit1
    normB = np.sqrt(f1B ** 2 + f2B ** 2)
    if normB < 1e-9:
        f1nB, f2nB = 1.0, 0.0
    else:
        f1nB, f2nB = f1B / normB, f2nB = f2B / normB
    qc.initialize([f1nB, f2nB], 1)

    # Optional entangling or swap test approach, here we do a partial approach
    # Just measure directly in the computational basis. The probability of
    # measuring the same state (00 or 11) can be used as a proxy for overlap.
    qc.barrier()
    qc.measure_all()

    return qc


def get_overlap(counts, shots):
    """
    Given the result counts from the distance_circuit,
    we measure overlap as p(00) + p(11).
    """
    p00 = counts.get('00', 0) / shots
    p11 = counts.get('11', 0) / shots
    return (p00 + p11)


###############################################################################
# 4) Put it all together for a QkNN classification:
#    For each test sample, measure overlap to each train sample using a quantum
#    circuit. Convert overlap to distance => distance = 1 - overlap, for instance.
#    Then pick k nearest neighbors by smallest distance and do a majority vote.
###############################################################################
def qknn_classify(x_test, x_train, y_train, k, shots=1024,
                  backend=None, noise_model=None):
    """
    x_test: [f1, f2] for a single test sample
    x_train: array of shape (num_train, 2)
    y_train: corresponding training labels
    k: number of neighbors
    shots: number of shots for each circuit
    backend: Qiskit backend (simulator)
    noise_model: Qiskit noise model or None
    """
    # Build one circuit per training sample
    circuits = []
    for i, train_pt in enumerate(x_train):
        circ = build_distance_circuit(x_test[0], x_test[1], train_pt[0], train_pt[1])
        circ = transpile(circ, backend=backend)
        circuits.append(circ)

    # Run all circuits as a single job
    job = backend.run(circuits, shots=shots, noise_model=noise_model)
    results = job.result()

    # For each circuit, extract overlap => distance
    distances = []
    for i, train_pt in enumerate(x_train):
        counts = results.get_counts(i)
        overlap = get_overlap(counts, shots)
        dist = 1.0 - overlap  # simple definition
        distances.append((dist, y_train[i]))

    # Sort by distance ascending
    distances.sort(key=lambda x: x[0])

    # Take the top k
    top_k = distances[:k]
    # majority vote
    neighbors_labels = [x[1] for x in top_k]
    vote_label = Counter(neighbors_labels).most_common(1)[0][0]
    return vote_label


def qknn_predict(X_test, X_train, y_train, k, backend, noise_model=None):
    """
    Vectorized classify for entire test set.
    Returns an array of predicted labels.
    """
    y_pred = []
    for xt in X_test:
        pred_label = qknn_classify(xt, X_train, y_train, k,
                                   shots=1024, backend=backend, noise_model=noise_model)
        y_pred.append(pred_label)
    return np.array(y_pred)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


###############################################################################
# 5) Main experiment loop:
#    - We will run for k=1..(test set size)
#    - We'll do it once with no noise, and once with T1 noise
###############################################################################

# Prepare the simulator
simulator = Aer.get_backend("aer_simulator")

# Sweep k from 1 to num_test
k_values = list(range(1, num_test + 1))

# Store accuracies
accs_no_noise = []
accs_t1_noise = []

###############################################################################
# NO NOISE run
###############################################################################
print("\nRunning QkNN with NO NOISE ...")
for k in k_values:
    y_pred_k = qknn_predict(X_test, X_train, y_train, k, backend=simulator, noise_model=None)
    acc_k = accuracy_score(y_test, y_pred_k)
    accs_no_noise.append(acc_k)
print("Accuracies (no noise):")
for k, acc in zip(k_values, accs_no_noise):
    print(f"  k={k}: acc={acc:.3f}")

###############################################################################
# WITH T1 NOISE run
#   We define a simple noise model that includes a T1 relaxation error
#   for each qubit.
###############################################################################
print("\nRunning QkNN WITH T1 NOISE ...")

# Example T1 = 50 microseconds, T2 = 70 microseconds for demonstration
# dt on many simulators is ~0.222222us, but let's keep it simple
# We'll define an approximate error for a single gate
T1 = 50e3  # 50 microseconds in ns
T2 = 70e3  # 70 microseconds in ns
gate_time_ns = 50  # single qubit gate ~50 ns

# Build relaxation errors for 1-qubit gate
# Probability p_reset or p_meas is small,
# but for demonstration, let's just do thermal relaxation error for single qubit
error_1q = thermal_relaxation_error(t1=T1, t2=T2, time=gate_time_ns, excited_state_population=0)

noise_model = NoiseModel()
# Add error to all single-qubit instructions
noise_model.add_all_qubit_quantum_error(error_1q, ["x", "u3", "u2", "u1", "cx", "initialize"])

accs_t1_noise = []
for k in k_values:
    y_pred_k = qknn_predict(X_test, X_train, y_train, k, backend=simulator, noise_model=noise_model)
    acc_k = accuracy_score(y_test, y_pred_k)
    accs_t1_noise.append(acc_k)
print("Accuracies (with T1 noise):")
for k, acc in zip(k_values, accs_t1_noise):
    print(f"  k={k}: acc={acc:.3f}")

###############################################################################
# 6) Plot results
###############################################################################
plt.figure(figsize=(8, 5))
plt.plot(k_values, accs_no_noise, marker='o', label='No Noise')
plt.plot(k_values, accs_t1_noise, marker='o', label='T1 Noise')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('QkNN on IRIS (2D subset) - Accuracy vs k')
plt.legend()
plt.grid(True)
plt.show()
