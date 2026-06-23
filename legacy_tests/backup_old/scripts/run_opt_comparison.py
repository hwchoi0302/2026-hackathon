"""
Extended mock simulation comparing:
  - 1st-order vs 2nd-order Trotter
  - Optimization level 1 vs 3
  - With and without ZNE
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Pauli
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer.noise import NoiseModel
from qiskit_aer import AerSimulator
import mitiq

from src.heisenberg import (
    get_heisenberg_hamiltonian, exact_evolution,
    build_first_order_trotter, build_second_order_trotter
)
from src.error_mitigation import apply_zne

# ── Constants ──────────────────────────────────────────────────────────────────
N = 4; J = 1.0; h = 0.5; T = 1.0; R = 5; SHOTS = 8192
os.makedirs('results', exist_ok=True)

# ── Exact reference ───────────────────────────────────────────────────────────
H = get_heisenberg_hamiltonian(N, J, h)
exact_U = exact_evolution(H, T)
init_sv = Statevector(QuantumCircuit(N))  # |0000>
init_qc = QuantumCircuit(N); init_qc.x(0); init_qc.x(2)
init_data = Statevector(init_qc).data
exact_sv_data = exact_U @ init_data
exact_val = float(Statevector(exact_sv_data).expectation_value(
    Pauli('I'*(N-1)+'Z')).real)
print(f"Exact <Z_0>: {exact_val:.4f}")

# ── Backend setup ─────────────────────────────────────────────────────────────
backend = FakeBrisbane()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)

def executor(circuit, opt_level=1):
    """Run circuit on noisy simulator and return <Z_0>."""
    c = circuit.copy()
    c.measure_all()
    tc = transpile(c, backend=backend, optimization_level=opt_level)
    counts = sim.run(tc, shots=SHOTS).result().get_counts()
    total = sum(counts.values())
    return sum((1. if b[-1]=='0' else -1.) * cnt for b, cnt in counts.items()) / total

# ── Experiments ───────────────────────────────────────────────────────────────
results = {"exact": exact_val, "experiments": []}

configs = [
    ("1st-order", build_first_order_trotter,  1),
    ("1st-order", build_first_order_trotter,  3),
    ("2nd-order", build_second_order_trotter, 1),
    ("2nd-order", build_second_order_trotter, 3),
]

for trotter_name, trotter_fn, opt_lvl in configs:
    label = f"{trotter_name} opt_lvl={opt_lvl}"
    print(f"\n── {label} ──")

    # Build base circuit
    qc = QuantumCircuit(N); qc.x(0); qc.x(2)
    qc.compose(trotter_fn(N, J, h, T, R), inplace=True)

    # Circuit stats (transpiled)
    t_qc = transpile(qc.copy().measure_all() if False else qc.copy(),
                     backend=backend, optimization_level=opt_lvl)
    depth = t_qc.depth()
    ecr   = t_qc.count_ops().get('ecr', 0)
    print(f"  Transpiled depth={depth}, ECR={ecr}")

    # Unmitigated
    def exec_fn(circ): return executor(circ, opt_level=opt_lvl)
    unmit = exec_fn(qc)
    print(f"  Unmitigated <Z_0>: {unmit:.4f}")

    # ZNE Richardson
    try:
        zne_rich = apply_zne(qc, exec_fn, scale_factors=[1.0, 2.0, 3.0])
        print(f"  ZNE Richardson  : {zne_rich:.4f}")
    except Exception as e:
        zne_rich = unmit
        print(f"  ZNE failed: {e}")

    results["experiments"].append({
        "label": label, "trotter_order": trotter_name,
        "opt_level": opt_lvl, "depth": depth, "ecr": ecr,
        "unmitigated": float(unmit), "zne_richardson": float(zne_rich),
        "error_unmitigated": abs(exact_val - unmit),
        "error_zne": abs(exact_val - zne_rich),
    })

with open('results/circuit_opt_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nSaved: results/circuit_opt_results.json")

# ── Plots ─────────────────────────────────────────────────────────────────────
exps = results["experiments"]
labels  = [f"{e['trotter_order']}\nopt={e['opt_level']}" for e in exps]
unmits  = [e['unmitigated']    for e in exps]
znes    = [e['zne_richardson'] for e in exps]
depths  = [e['depth']          for e in exps]
ecrs    = [e['ecr']            for e in exps]

x = np.arange(len(labels))
w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -- Left: <Z_0> values
ax = axes[0]
b1 = ax.bar(x - w/2, unmits, w, label='Unmitigated', color='#FF7043', edgecolor='black')
b2 = ax.bar(x + w/2, znes,   w, label='ZNE (Richardson)', color='#66BB6A', edgecolor='black')
ax.axhline(exact_val, color='#2196F3', linestyle='--', linewidth=1.5, label=f'Exact = {exact_val:.4f}')
for bar, v in zip(list(b1)+list(b2), unmits+znes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(r'$\langle Z_0\rangle$', fontsize=12)
ax.set_title('Expectation Value\n(FakeBrisbane, r=5)', fontsize=11)
ax.set_ylim(-0.1, 0.65); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.35)

# -- Right: Circuit depth and ECR count
ax2 = axes[1]
color_d = '#5C6BC0'; color_e = '#EF5350'
ax2r = ax2.twinx()
b3 = ax2.bar(x - w/2, depths, w, label='Transpiled Depth', color=color_d, alpha=0.8, edgecolor='black')
b4 = ax2r.bar(x + w/2, ecrs,  w, label='ECR Gate Count',   color=color_e, alpha=0.8, edgecolor='black')
for bar, v in zip(b3, depths):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(v),
             ha='center', va='bottom', fontsize=8.5, color=color_d)
for bar, v in zip(b4, ecrs):
    ax2r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(v),
              ha='center', va='bottom', fontsize=8.5, color=color_e)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Transpiled Depth', color=color_d, fontsize=11)
ax2r.set_ylabel('ECR Gate Count', color=color_e, fontsize=11)
ax2.set_title('Circuit Complexity\n(After Transpilation)', fontsize=11)
lines1, lab1 = ax2.get_legend_handles_labels()
lines2, lab2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1+lines2, lab1+lab2, fontsize=9)
ax2.grid(axis='y', alpha=0.25)

plt.suptitle('1st vs 2nd Order Trotter + Optimization Level Comparison (FakeBrisbane, r=5)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/opt_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/opt_comparison.png")
