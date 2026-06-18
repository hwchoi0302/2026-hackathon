"""
Quick diagnostic script for Heisenberg Trotter implementation.
Runs without noisy backend to quickly verify correctness.
"""
import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, Operator

from src.heisenberg import (
    get_heisenberg_hamiltonian,
    exact_evolution,
    build_first_order_trotter,
    build_second_order_trotter,
)

N = 4
J = 1.0
h = 0.5
T = 1.0

print("=== Hamiltonian check ===")
H = get_heisenberg_hamiltonian(N, J, h)
print(f"  Hermitian: {np.allclose(H, H.conj().T)}")
print(f"  Eigenvalues (first 4): {np.linalg.eigvalsh(H)[:4].round(3)}")

print("\n=== Exact unitary check ===")
exact_U = exact_evolution(H, T)
print(f"  Unitary: {np.allclose(exact_U @ exact_U.conj().T, np.eye(2**N), atol=1e-10)}")

print("\n=== Initial state: |1010> ===")
# Qiskit label '1010': qubit 0 = rightmost char = '0'? No:
# from_label('1010') -> q0=|1>, q1=|0>, q2=|1>, q3=|0>
# but Statevector stores q0 as least-significant bit
# Same as x(0).x(2) in circuit: qubit 0 and 2 flipped
init_qc = QuantumCircuit(N)
init_qc.x(0)
init_qc.x(2)
init_sv = Statevector(init_qc)
init_from_label = Statevector.from_label('0101')  # qubit3..0: 0,1,0,1 -> qubits 1,3 flipped
print(f"  |1010> from circuit vs from_label('0101'): {np.allclose(init_sv.data, init_from_label.data)}")

print("\n=== Evolving with exact unitary ===")
# Method 1: use Operator - correct qubit mapping
exact_sv_op = init_sv.evolve(Operator(exact_U))
# Method 2: numpy directly
exact_state_np = exact_U @ init_sv.data
print(f"  Norm of exact state: {np.linalg.norm(exact_sv_op.data):.6f}")
print(f"  Op vs numpy same: {np.allclose(exact_sv_op.data, exact_state_np)}")

print("\n=== Trotter fidelities (using Operator for exact) ===")
for r in [1, 2, 5, 10, 20]:
    qc1 = QuantumCircuit(N)
    qc1.x(0); qc1.x(2)
    qc1.compose(build_first_order_trotter(N, J, h, T, r), inplace=True)
    sv1 = Statevector(qc1)
    f1 = state_fidelity(exact_sv_op, sv1)

    qc2 = QuantumCircuit(N)
    qc2.x(0); qc2.x(2)
    qc2.compose(build_second_order_trotter(N, J, h, T, r), inplace=True)
    sv2 = Statevector(qc2)
    f2 = state_fidelity(exact_sv_op, sv2)
    print(f"  r={r:2d} | 1st-order: {f1:.4f} | 2nd-order: {f2:.4f}")

print("\n=== Trotter fidelities (using numpy direct for exact) ===")
for r in [1, 2, 5, 10, 20]:
    qc1 = QuantumCircuit(N)
    qc1.x(0); qc1.x(2)
    qc1.compose(build_first_order_trotter(N, J, h, T, r), inplace=True)
    sv1 = Statevector(qc1).data

    qc2 = QuantumCircuit(N)
    qc2.x(0); qc2.x(2)
    qc2.compose(build_second_order_trotter(N, J, h, T, r), inplace=True)
    sv2 = Statevector(qc2).data

    # Fidelity = |<exact|trotter>|^2
    f1 = abs(exact_state_np.conj() @ sv1)**2
    f2 = abs(exact_state_np.conj() @ sv2)**2
    print(f"  r={r:2d} | 1st-order: {f1:.4f} | 2nd-order: {f2:.4f}")
