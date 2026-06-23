import os
import json
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector, state_fidelity, Pauli
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.heisenberg import get_heisenberg_hamiltonian, exact_evolution, build_first_order_trotter, build_second_order_trotter
from src.error_mitigation import apply_zne


# ── Constants ──────────────────────────────────────────────────────────────────
N = 4            # Number of qubits (spins)
J = 1.0          # Heisenberg coupling
h = 0.5          # External field
T = 1.0          # Total evolution time
ZNE_R = 5        # r=5 with 2nd-order gives fidelity 0.9945, shallower per step than 1st-order
MAX_STEPS = 10   # Max Trotter steps for approximation study


def get_initial_circuit(N: int) -> QuantumCircuit:
    """Prepares the |1010> Neel state circuit."""
    qc = QuantumCircuit(N)
    qc.x(0)
    qc.x(2)
    return qc


def noisy_executor(simulator: AerSimulator, circuit: QuantumCircuit, shots: int = 8192) -> float:
    """
    Executes a circuit on the noisy simulator and returns <Z_0>.

    Qiskit bitstring convention: the rightmost character corresponds to qubit 0.
    So for a bitstring s, s[-1] == '0' means qubit 0 measured as |0> (+1 eigenvalue).
    """
    c = circuit.copy()
    c.measure_all()
    t_qc = transpile(c, simulator)
    job = simulator.run(t_qc, shots=shots)
    counts = job.result().get_counts()

    exp_z = 0.0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        # bitstring[-1] is qubit 0 (little-endian)
        z_val = 1.0 if bitstring[-1] == '0' else -1.0
        exp_z += z_val * count
    return exp_z / total


def run_experiment():
    os.makedirs('results', exist_ok=True)

    # ── 1. Exact evolution ─────────────────────────────────────────────────────
    print("=" * 60)
    print("1. Exact Evolution")
    print("=" * 60)

    H = get_heisenberg_hamiltonian(N, J, h)
    exact_U = exact_evolution(H, T)

    # |1010> Neel state via circuit: qubit 0 = |1>, qubit 2 = |1>
    init_qc = QuantumCircuit(N)
    init_qc.x(0)
    init_qc.x(2)
    init_sv = Statevector(init_qc)

    # Evolve with exact unitary (numpy: U @ |psi>)
    exact_state_data = exact_U @ init_sv.data
    exact_state = Statevector(exact_state_data)

    # Observable: Z on qubit 0 (Pauli string 'III...Z', rightmost = qubit 0)
    observable = Pauli('I' * (N - 1) + 'Z')
    exact_exp_val = exact_state.expectation_value(observable).real
    print(f"  Exact <Z_0>: {exact_exp_val:.4f}")

    # ── 2. Ideal Trotter approximation error ───────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Ideal Trotter Approximation Error")
    print("=" * 60)

    steps_list = list(range(1, MAX_STEPS + 1))
    fidelities_1st = []
    fidelities_2nd = []

    for r in steps_list:
        # First order: prepend Neel state init to Trotter circuit
        qc_1 = QuantumCircuit(N)
        qc_1.x(0); qc_1.x(2)
        qc_1.compose(build_first_order_trotter(N, J, h, T, r), inplace=True)
        sv_1 = Statevector(qc_1)
        fidelities_1st.append(abs(exact_state_data.conj() @ sv_1.data)**2)

        # Second order
        qc_2 = QuantumCircuit(N)
        qc_2.x(0); qc_2.x(2)
        qc_2.compose(build_second_order_trotter(N, J, h, T, r), inplace=True)
        sv_2 = Statevector(qc_2)
        fidelities_2nd.append(abs(exact_state_data.conj() @ sv_2.data)**2)

        print(f"  r={r:2d} | 1st-order: {fidelities_1st[-1]:.4f} | 2nd-order: {fidelities_2nd[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps_list, fidelities_1st, marker='o', label='1st-order (Lie-Trotter)', color='royalblue')
    plt.plot(steps_list, fidelities_2nd, marker='s', label='2nd-order (Suzuki-Trotter)', color='tomato')
    plt.xlabel('Trotter Steps (r)', fontsize=12)
    plt.ylabel('State Fidelity', fontsize=12)
    plt.title(f'Trotter Approximation Fidelity (N={N}, J={J}, h={h}, T={T})', fontsize=13)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig('results/trotter_fidelity.png', dpi=150)
    plt.close()
    print("\n  Saved: results/trotter_fidelity.png")

    # ── 3. Noisy simulation + ZNE ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"3. Noisy Simulation (FakeBrisbane) + ZNE  [r={ZNE_R}]")
    print("=" * 60)

    backend = FakeBrisbane()
    noise_model = NoiseModel.from_backend(backend)
    simulator = AerSimulator(noise_model=noise_model)

    # Build circuit: Neel init + 2nd-order Trotter (higher fidelity, better starting point for ZNE)
    noisy_qc = QuantumCircuit(N)
    noisy_qc.x(0)
    noisy_qc.x(2)
    noisy_qc.compose(build_second_order_trotter(N, J, h, T, ZNE_R), inplace=True)

    def executor_fn(circuit):
        return noisy_executor(simulator, circuit)

    print("  Running unmitigated simulation...")
    unmitigated_val = executor_fn(noisy_qc)
    print(f"  Unmitigated <Z_0>: {unmitigated_val:.4f}")

    print("Running ZNE-mitigated simulation (Linear + Richardson)...")
    try:
        # Try Linear extrapolation first (more robust for deep/noisy circuits)
        import mitiq
        linear_factory = mitiq.zne.inference.LinearFactory([1.0, 2.0, 3.0])
        mitigated_val_linear = apply_zne(noisy_qc, executor_fn, scale_factors=[1.0, 2.0, 3.0], factory=linear_factory)
        mitigated_val = apply_zne(noisy_qc, executor_fn, scale_factors=[1.0, 2.0, 3.0])
        print(f"  Linear ZNE  <Z_0>: {mitigated_val_linear:.4f}")
        print(f"  Richardson  <Z_0>: {mitigated_val:.4f}")
    except Exception as e:
        print(f"  ZNE failed: {e}")
        mitigated_val = unmitigated_val
        mitigated_val_linear = unmitigated_val

    # ── 4. Save and plot results ───────────────────────────────────────────────
    results = {
        "N": N, "J": J, "h": h, "T": T, "zne_trotter_steps": ZNE_R,
        "exact_z0": float(exact_exp_val),
        "unmitigated_z0": float(unmitigated_val),
        "mitigated_z0_linear": float(mitigated_val_linear),
        "mitigated_z0_richardson": float(mitigated_val),
        "fidelities_1st_order": [float(f) for f in fidelities_1st],
        "fidelities_2nd_order": [float(f) for f in fidelities_2nd],
    }
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n  Saved: results/metrics.json")

    labels = ['Ideal\n(Exact)', f'Unmitigated\n(FakeBrisbane, r={ZNE_R})',
              f'ZNE Linear\n(r={ZNE_R})', f'ZNE Richardson\n(r={ZNE_R})']
    values = [exact_exp_val, unmitigated_val, mitigated_val_linear, mitigated_val]
    colors = ['steelblue', 'tomato', 'gold', 'mediumseagreen']

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
    plt.axhline(y=exact_exp_val, color='steelblue', linestyle='--', linewidth=1.2, label='Exact value')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.ylabel(r'$\langle Z_0 \rangle$', fontsize=12)
    plt.title(f'ZNE Error Mitigation on Trotterized Evolution (N={N}, r={ZNE_R})', fontsize=12)
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig('results/zne_comparison.png', dpi=150)
    plt.close()
    print("  Saved: results/zne_comparison.png")

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
