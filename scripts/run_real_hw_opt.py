"""
Real IBM hardware experiment: compare 1st vs 2nd order Trotter
with optimization levels 1 and 3 on ibm_yonsei.

Uses saved credentials from previous run. If not saved, set:
  export IBM_IAM_APIKEY="..."
  export IBM_CRN="..."
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from src.heisenberg import build_first_order_trotter, build_second_order_trotter
from src.error_mitigation import apply_zne

# ── Parameters ─────────────────────────────────────────────────────────────────
N = 4; J = 1.0; h = 0.5; T = 1.0; R = 5
SHOTS = 8192
BACKEND_NAME = "ibm_yonsei"
EXACT = 0.48426

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
def real_executor(circuit, opt_level=1):
    """Runs circuit on real hardware and returns <Z_0>."""
    c = circuit.copy()
    c.measure_all()
    tc = transpile(c, backend=backend, optimization_level=opt_level)
    sampler = Sampler(backend)
    job = sampler.run([tc], shots=SHOTS)
    print(f"    Job {job.job_id()} submitted...")
    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    return sum((1. if b[-1]=='0' else -1.) * cnt for b, cnt in counts.items()) / total

# ── Run 4 configurations ──────────────────────────────────────────────────────
configs = [
    ("1st-order", build_first_order_trotter,  1),
    ("1st-order", build_first_order_trotter,  3),
    ("2nd-order", build_second_order_trotter, 1),
    ("2nd-order", build_second_order_trotter, 3),
]

results = {"exact": EXACT, "backend": BACKEND_NAME, "shots": SHOTS,
           "N": N, "J": J, "h": h, "T": T, "r": R, "experiments": []}

for trotter_name, trotter_fn, opt_lvl in configs:
    label = f"{trotter_name} opt={opt_lvl}"
    print(f"\n{'─'*55}\n{label}")

    qc = QuantumCircuit(N); qc.x(0); qc.x(2)
    qc.compose(trotter_fn(N, J, h, T, R), inplace=True)

    # Circuit stats
    t_qc = transpile(qc, backend=backend, optimization_level=opt_lvl)
    depth = t_qc.depth()
    ecr   = t_qc.count_ops().get('ecr', 0)
    print(f"  Transpiled depth={depth}, ECR={ecr}")

    def exec_fn(circ):
        return real_executor(circ, opt_level=opt_lvl)

    # Unmitigated
    print("  Running unmitigated job...")
    unmit = exec_fn(qc)
    print(f"  Unmitigated <Z_0>: {unmit:.4f}")

    # ZNE Richardson
    print("  Running ZNE jobs (x3 scale factors)...")
    try:
        zne_val = apply_zne(qc, exec_fn, scale_factors=[1.0, 2.0, 3.0])
        print(f"  ZNE Richardson  : {zne_val:.4f}")
    except Exception as e:
        zne_val = unmit
        print(f"  ZNE failed: {e}")

    results["experiments"].append({
        "label": label, "trotter_order": trotter_name,
        "opt_level": opt_lvl, "depth": depth, "ecr": ecr,
        "unmitigated": float(unmit),
        "zne_richardson": float(zne_val),
        "error_unmitigated": abs(EXACT - unmit),
        "error_zne": abs(EXACT - zne_val),
    })

with open('results/real_hw_opt_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("\nSaved: results/real_hw_opt_results.json")

# ── Plot ─────────────────────────────────────────────────────────────────────
exps   = results["experiments"]
labels = [f"{e['trotter_order']}\nopt={e['opt_level']}" for e in exps]
unmits = [e['unmitigated']    for e in exps]
znes   = [e['zne_richardson'] for e in exps]
depths = [e['depth']          for e in exps]
ecrs   = [e['ecr']            for e in exps]

x = np.arange(len(labels)); w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
b1 = ax.bar(x - w/2, unmits, w, label='Unmitigated',     color='#EC407A', edgecolor='black')
b2 = ax.bar(x + w/2, znes,   w, label='ZNE (Richardson)', color='#AB47BC', edgecolor='black')
ax.axhline(EXACT, color='#2196F3', linestyle='--', linewidth=1.5, label=f'Exact = {EXACT:.4f}')
for bar, v in zip(list(b1)+list(b2), unmits+znes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(r'$\langle Z_0\rangle$', fontsize=12)
ax.set_title(f'Real Hardware: {BACKEND_NAME}\n(r={R}, N={N})', fontsize=11)
ax.set_ylim(-0.3, 0.65); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.35)

ax2 = axes[1]; ax2r = ax2.twinx()
cd = '#5C6BC0'; ce = '#EF5350'
b3 = ax2.bar(x - w/2, depths, w, color=cd, alpha=0.8, edgecolor='black', label='Depth')
b4 = ax2r.bar(x + w/2, ecrs,  w, color=ce, alpha=0.8, edgecolor='black', label='ECR')
for bar, v in zip(b3, depths):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(v),
             ha='center', va='bottom', fontsize=8.5, color=cd)
for bar, v in zip(b4, ecrs):
    ax2r.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(v),
              ha='center', va='bottom', fontsize=8.5, color=ce)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Transpiled Depth', color=cd, fontsize=11)
ax2r.set_ylabel('ECR Gate Count', color=ce, fontsize=11)
ax2.set_title('Circuit Complexity After Transpilation', fontsize=11)
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, fontsize=9)
ax2.grid(axis='y', alpha=0.25)

plt.suptitle(f'Real Hardware ({BACKEND_NAME}): 1st vs 2nd Trotter + Circuit Optimization',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/real_hw_opt_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/real_hw_opt_comparison.png")
