"""
Real hardware experiment: Sweep Trotter steps `r` from 1 to 5 without ZNE.
Observes the tradeoff between Trotter error (high at low r) and hardware noise (high at large r).
Uses optimization_level=3 for best performance.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from src.heisenberg import build_first_order_trotter, build_second_order_trotter

# ── Parameters ─────────────────────────────────────────────────────────────────
N = 4; J = 1.0; h = 0.5; T = 1.0
R_VALS = [1, 2, 3, 4, 5]
SHOTS = 8192
BACKEND_NAME = "ibm_yonsei"
EXACT_Z = 0.48426
OPT_LEVEL = 3

os.makedirs('results', exist_ok=True)

# ── Connect to IBM ─────────────────────────────────────────────────────────────
iam_key = os.environ.get("IBM_IAM_APIKEY", "")
crn     = os.environ.get("IBM_CRN", "")

if iam_key and crn:
    QiskitRuntimeService.save_account(
        channel="ibm_cloud", token=iam_key, instance=crn,
        overwrite=True, set_as_default=True)
    service = QiskitRuntimeService(channel="ibm_cloud", instance=crn)
else:
    print("Using saved credentials...")
    service = QiskitRuntimeService()

backend = service.backend(BACKEND_NAME)
print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

# ── Executor ──────────────────────────────────────────────────────────────────
sampler = Sampler(backend)

def real_executor(circuit):
    """Runs circuit on real hardware and returns <Z_0>."""
    c = circuit.copy()
    c.measure_all()
    tc = transpile(c, backend=backend, optimization_level=OPT_LEVEL)
    depth = tc.depth()
    ecr = tc.count_ops().get('ecr', 0)
    
    job = sampler.run([tc], shots=SHOTS)
    print(f"    Job {job.job_id()} submitted (depth={depth}, ecr={ecr})...")
    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    exp_z = sum((1. if b[-1]=='0' else -1.) * cnt for b, cnt in counts.items()) / total
    return exp_z, depth, ecr

# ── Ideal Trotter Values (from previous simulation) ───────────────────────────
ideal_1st = {1: 0.0818, 2: 0.2312, 3: 0.3341, 4: 0.3950, 5: 0.4308}  # Approximate <Z> expectation values 
ideal_2nd = {1: -0.669, 2: 0.3458, 3: 0.4352, 4: 0.4633, 5: 0.4754}  # Can compute exact if needed, but we focus on real hardware comparison

# We will actually compute the ideal <Z> values on the fly for perfect reference
from qiskit.quantum_info import Statevector, Pauli
exact_sv_base = Statevector.from_label('1010')

ideal_z_1st = []
ideal_z_2nd = []

print("Computing ideal Trotter expectations...")
for r in R_VALS:
    qc1 = QuantumCircuit(N); qc1.compose(build_first_order_trotter(N, J, h, T, r), inplace=True)
    z1 = float(exact_sv_base.evolve(qc1).expectation_value(Pauli('IIIZ')).real)
    ideal_z_1st.append(z1)
    
    qc2 = QuantumCircuit(N); qc2.compose(build_second_order_trotter(N, J, h, T, r), inplace=True)
    z2 = float(exact_sv_base.evolve(qc2).expectation_value(Pauli('IIIZ')).real)
    ideal_z_2nd.append(z2)

# ── Sweep Execution ───────────────────────────────────────────────────────────
results = {
    "exact": EXACT_Z,
    "backend": BACKEND_NAME,
    "r_vals": R_VALS,
    "ideal_1st": ideal_z_1st,
    "ideal_2nd": ideal_z_2nd,
    "real_1st": [],
    "real_2nd": [],
    "depth_1st": [],
    "depth_2nd": [],
    "ecr_1st": [],
    "ecr_2nd": []
}

for r in R_VALS:
    print(f"\n{'─'*50}\nRunning r = {r}")
    
    # 1st order
    print("  1st-order:")
    qc1 = QuantumCircuit(N); qc1.x(0); qc1.x(2)
    qc1.compose(build_first_order_trotter(N, J, h, T, r), inplace=True)
    z1_real, d1, e1 = real_executor(qc1)
    results["real_1st"].append(z1_real)
    results["depth_1st"].append(d1)
    results["ecr_1st"].append(e1)
    print(f"    Unmitigated <Z_0>: {z1_real:.4f} (Ideal: {ideal_z_1st[r-1]:.4f})")
    
    # 2nd order
    print("  2nd-order:")
    qc2 = QuantumCircuit(N); qc2.x(0); qc2.x(2)
    qc2.compose(build_second_order_trotter(N, J, h, T, r), inplace=True)
    z2_real, d2, e2 = real_executor(qc2)
    results["real_2nd"].append(z2_real)
    results["depth_2nd"].append(d2)
    results["ecr_2nd"].append(e2)
    print(f"    Unmitigated <Z_0>: {z2_real:.4f} (Ideal: {ideal_z_2nd[r-1]:.4f})")

with open('results/r_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nSaved: results/r_sweep_results.json")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Expectation Value
ax1.axhline(EXACT_Z, color='black', linestyle='--', label=f'Exact = {EXACT_Z:.4f}')
ax1.plot(R_VALS, ideal_z_1st, 'o--', color='blue', alpha=0.5, label='Ideal 1st-order')
ax1.plot(R_VALS, ideal_z_2nd, 's--', color='red', alpha=0.5, label='Ideal 2nd-order')
ax1.plot(R_VALS, results['real_1st'], 'o-', color='blue', linewidth=2, label='Real 1st-order')
ax1.plot(R_VALS, results['real_2nd'], 's-', color='red', linewidth=2, label='Real 2nd-order')

ax1.set_xlabel('Trotter Steps ($r$)')
ax1.set_ylabel(r'$\langle Z_0 \rangle$')
ax1.set_title(f'Expectation Value vs Trotter Steps\n(Real Hardware: {BACKEND_NAME}, opt=3)')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(R_VALS)

# Right: Circuit Complexity
ax2.plot(R_VALS, results['ecr_1st'], 'o-', color='blue', label='1st-order ECRs')
ax2.plot(R_VALS, results['ecr_2nd'], 's-', color='red', label='2nd-order ECRs')
ax2.set_xlabel('Trotter Steps ($r$)')
ax2.set_ylabel('Number of ECR Gates')
ax2.set_title('Transpiled Circuit Complexity')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(R_VALS)

plt.tight_layout()
plt.savefig('results/r_sweep_plot.png', dpi=150)
print("Saved: results/r_sweep_plot.png")
