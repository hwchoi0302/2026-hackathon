"""
Compute EXACT time evolution of <Z_0> via matrix exponentiation of the XYZ Hamiltonian.
This serves as the theoretical 'ground truth' to compare against Trotter approximations.
"""
import os, sys, json
import numpy as np
from scipy.linalg import expm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian

# ── Parameters (must match run_harness.py) ──
N = 4
JX, JY, JZ, h = 1.0, 0.8, 0.5, 0.5
T_MAX = 2.0
DT = 0.1

def main():
    hamiltonian = get_xyz_hamiltonian(N, JX, JY, JZ, h)
    
    # Convert SparsePauliOp to dense matrix
    H_matrix = hamiltonian.to_matrix()
    
    # Eigenvalue decomposition for spectrum info
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    print("=== Exact Eigenvalues of H ===")
    for i, ev in enumerate(eigenvalues):
        print(f"  E_{i} = {ev:.6f}")
    print(f"\n  Ground state energy: E_0 = {eigenvalues[0]:.6f}")
    print(f"  First excited state: E_1 = {eigenvalues[1]:.6f}")
    print(f"  Energy gap (E_1 - E_0): {eigenvalues[1] - eigenvalues[0]:.6f}")
    
    # Compute unique energy differences (positive only)
    energy_diffs = []
    for i in range(len(eigenvalues)):
        for j in range(i+1, len(eigenvalues)):
            energy_diffs.append(eigenvalues[j] - eigenvalues[i])
    energy_diffs = sorted(set(np.round(energy_diffs, 8)))
    
    # Convert to frequencies (omega = dE, f = omega / (2*pi))
    print("\n=== Energy Differences (ΔE = ω) and Frequencies (f = ω/2π) ===")
    for de in energy_diffs[:10]:
        print(f"  ΔE = {de:.6f}  →  f = {de/(2*np.pi):.6f}")
    
    # Initial state: Neel state |0101> 
    # In Qiskit little-endian convention, |0101> means qubit 0=1, qubit 1=0, qubit 2=1, qubit 3=0
    # Binary: q3 q2 q1 q0 = 0 1 0 1 -> decimal = 5
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[5] = 1.0  # |0101> in little-endian
    
    # Observable: Z_0 = I ⊗ I ⊗ I ⊗ Z (little-endian)
    from qiskit.quantum_info import SparsePauliOp
    Z0 = SparsePauliOp("IIIZ").to_matrix()
    
    # Time evolution
    times = np.arange(0.0, T_MAX + DT, DT)
    exact_values = []
    
    print("\n=== Exact Time Evolution <Z_0(t)> ===")
    for t in times:
        if t == 0.0:
            val = np.real(psi0.conj() @ Z0 @ psi0)
        else:
            U = expm(-1j * H_matrix * t)
            psi_t = U @ psi0
            val = np.real(psi_t.conj() @ Z0 @ psi_t)
        exact_values.append(float(val))
        print(f"  t={t:.2f} | <Z_0> = {val:.6f}")
    
    # Overlap of initial state with eigenstates
    print("\n=== Overlap |<E_n|ψ(0)>|² ===")
    overlaps = []
    for i in range(len(eigenvalues)):
        overlap = np.abs(eigenvectors[:, i].conj() @ psi0)**2
        overlaps.append(float(overlap))
        if overlap > 1e-6:
            print(f"  |<E_{i}|ψ(0)>|² = {overlap:.6f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    data = {
        "eigenvalues": [float(e) for e in eigenvalues],
        "energy_diffs": [float(d) for d in energy_diffs],
        "frequencies_hz": [float(d/(2*np.pi)) for d in energy_diffs],
        "ground_state_energy": float(eigenvalues[0]),
        "overlaps": overlaps,
        "times": list(times),
        "exact_Z0": exact_values
    }
    path = "results/exact_diagonalization.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved exact results to: {path}")

if __name__ == "__main__":
    main()
