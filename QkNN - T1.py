"""
QkNN_T1.py
----------
Simple demonstration of a quantum kNN approach on IRIS data (2D subset).
Uses Qiskit Aer simulator with an optional T1 noise model.
"""

import numpy as np
import matplotlib.pyplot as plt

# Qiskit
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers.aer.noise import NoiseModel, thermal_relaxation_error

# For IRIS dataset, train/test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

###############################################################################
# 1) Load IRIS dataset, 80/20 train/test split
###############################################################################
iris = load_iris()
X = iris.data       # shape (150, 4)
y = iris.target     # shape (150, )
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# For simplicity, let's use only the first 2 features for amplitude encoding
X = X[:, :2]

# Normalize data to [0,1] range
scaler = MinMaxScaler()
X = scaler.fit_transform(X)  # shape (150, 2) now in [0,1]

# Create 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=42
)

num_train = len(X_train)
num_test = len(X_test)
print(f"Training set size: {num_train}, Test set size: {num_test}")

###############################################################################
# 2) Function to build a circuit encoding a single 2D point
###############################################################################
def encode_point_amplitude(f1, f2):
    """
    Build a 1-qubit circuit encoding the point (f1, f2) as amplitudes.
    We'll assume f1^2 + f2^2 <= 1; otherwise re-normalize.
    """
    norm = np.sqrt(f1**2 + f2**2)
    if norm < 1e-9:
        f1n, f2n = 1.0, 0.0
    else:
        f1n, f2n = f1 / norm, f2 / norm

    qc = QuantumCircuit(1, name="encode")
    qc.initialize([f1n, f2n], 0)
    return qc

###############################################################################
# 3) Build a circuit that measures "distance" or "overlap" between two points
###############################################################################
def build_distance_circuit(f1A, f2A, f1B, f2B):
    """
    Return a 2-qubit circuit that attempts to measure the overlap
    between point A and point B.
    We'll measure p(00) + p(11) as an overlap measure.
    """
    qc = QuantumCircuit(2)

    # Encode A on qubit0
    normA = np.sqrt(f1A**2 + f2A**2)
    if normA < 1e-9:
        f1nA, f2nA = 1.0, 0.0
    else:
        f1nA, f2nA = f1A / normA, f2A / normA
    qc.initialize([f1nA, f2nA], 0)

    # Encode B on qubit1
    normB = np.sqrt(f1B**2 + f2B**2)
    if normB < 1e-9:
        f1nB, f2nB = 1.0, 0.0
    else:
        f1nB, f2nB = f1B / normB, f2B / normB  # <-- FIXED the syntax here!

    qc.initialize([f1nB, f2nB], 1)

    # Barrier for clarity, then measure all
    qc.barrier()
    qc.measure_all()

    return qc

def get_overlap(counts, shots):
    """
    Overlap = Probability(00 or 11).
    """
    p00 = counts.get('00', 0) / shots
    p11 = counts.get('11', 0) / shots
    return p00 + p11

###############################################################################
# 4) QkNN classification logic
###############################################################################
def qknn_classify(x_test, x_train, y_train, k, shots=1024,
                  backend=None, noise_model=None):
    """
    Classify a single test point x_test using a quantum 'distance' to each training point.
    """
    # Build one circuit per training sample
    circuits = []
    for i, train_pt in enumerate(x_train):
        circ = build_distance_circuit(x_test[0], x_test[1],
                                      train_pt[0], train_pt[1])
        circ = transpile(circ, backend=backend)
        circuits.append(circ)

    # Execute all circuits as one job
    job = backend.run(circuits, shots=shots, noise_model=noise_model)
    results = job.result()

    # For each circuit, compute overlap => distance
    distances = []
    for i, train_pt in enumerate(x_train):
        counts = results.get_counts(i)
        overlap = get_overlap(counts, shots)
        dist = 1.0 - overlap  # 1 - overlap
        distances.append((dist, y_train[i]))

    # Sort by distance ascending
    distances.sort(key=lambda x: x[0])
    # Grab top k
    top_k = distances[:k]
    neighbor_labels = [t[1] for t in top_k]
    # Majority vote
    vote_label = Counter(neighbor_labels).most_common(1)[0][0]
    return vote_label

def qknn_predict(X_test, X_train, y_train, k, backend,
                 noise_model=None, shots=1024):
    """
    Vectorized classify for entire test set.
    """
    preds = []
    for xt in X_test:
        label = qknn_classify(xt, X_train, y_train, k,
                              shots=shots,
                              backend=backend,
                              noise_model=noise_model)
        preds.append(label)
    return np.array(preds)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

###############################################################################
# 5) Main experiment loop: run for k=1..(test set size),
#    Compare No Noise vs T1 Noise
###############################################################################
simulator = Aer.get_backend("aer_simulator")

k_values = list(range(1, num_test+1))
accs_no_noise = []
accs_t1_noise = []

##############################
# NO NOISE
##############################
print("\nRunning QkNN (No Noise)...")
for k in k_values:
    y_pred = qknn_predict(X_test, X_train, y_train, k,
                          backend=simulator, noise_model=None)
    acc_k = accuracy_score(y_test, y_pred)
    accs_no_noise.append(acc_k)
    print(f" k={k}, accuracy={acc_k:.3f}")

##############################
# WITH T1 NOISE
##############################
print("\nRunning QkNN (T1 Noise)...")

# Example T1/T2 times
T1 = 50e3  # 50 microseconds in ns
T2 = 70e3  # 70 microseconds in ns
gate_time_ns = 50  # approximate gate time in ns

# Build T1 relaxation error for single-qubit gates
error_1q = thermal_relaxation_error(
    t1=T1, t2=T2, time=gate_time_ns, excited_state_population=0
)

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1q,
                                        ["x", "u3", "u2", "u1",
                                         "cx", "initialize"])

for k in k_values:
    y_pred = qknn_predict(X_test, X_train, y_train, k,
                          backend=simulator, noise_model=noise_model)
    acc_k = accuracy_score(y_test, y_pred)
    accs_t1_noise.append(acc_k)
    print(f" k={k}, accuracy={acc_k:.3f}")

##############################
# 6) Plot the results
##############################
plt.figure(figsize=(8,5))
plt.plot(k_values, accs_no_noise, marker='o', label='No Noise')
plt.plot(k_values, accs_t1_noise, marker='o', label='T1 Noise')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('QkNN on IRIS (2D subset) - Accuracy vs k')
plt.legend()
plt.grid(True)
plt.show()
