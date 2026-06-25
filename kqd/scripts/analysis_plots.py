"""
종합 분석 스크립트:
1. Trotter step 수에 따른 KQD 에너지 수렴 및 회로 깊이/2Q 게이트 비교
2. 표준 Trotterization 시간 동역학 데이터에 대한 고전적 FFT 분석으로 에너지 추출
3. 표준 Trotterization 회로 메트릭 수집
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian
from src.circuit_builder import build_trotter_circuit
from src.execution import get_estimator
from src.kqd import (
    solve_regularized_gen_eig,
    is_vacuum_eigenstate,
    build_kqd_template_circuit
)

# ============================================================
# 공통 설정
# ============================================================
N = 4
JX, JY, JZ, h_val = 1.0, 1.0, 0.5, 0.5
H_op = get_xyz_hamiltonian(N, JX, JY, JZ, h_val)
exact_eigvals = np.sort(np.linalg.eigvalsh(H_op.to_matrix()))
exact_gnd = exact_eigvals[0]
print(f"Exact ground state energy: {exact_gnd:.6f}")
print(f"Exact eigenvalues (first 8): {exact_eigvals[:8]}")

os.makedirs('results', exist_ok=True)

def get_neel_state(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        qc.x(i)
    return qc

# dt 계산
def get_kqd_dt(H_op, n_qubits):
    from qiskit.quantum_info import Pauli
    single_particle_H = np.zeros((n_qubits, n_qubits), dtype=complex)
    for i in range(n_qubits):
        for j in range(i + 1):
            for p, coeff in H_op.to_list():
                p_rev = p[::-1]
                x_bits = [k for k, char in enumerate(p_rev) if char in ('X', 'Y')]
                z_bits = [k for k, char in enumerate(p_rev) if char in ('Z', 'Y')]
                diff_set = set([i, j])
                if set(x_bits) == diff_set or (i == j and len(x_bits) == 0):
                    num_y = sum(1 for char in p_rev if char == 'Y')
                    z_sign = 1
                    for z_bit in z_bits:
                        if z_bit == i:
                            z_sign *= -1
                    sgn = ((-1j) ** num_y) * z_sign
                else:
                    sgn = 0
                single_particle_H[i, j] += sgn * coeff
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            single_particle_H[i, j] = np.conj(single_particle_H[j, i])
    norm = np.linalg.norm(single_particle_H, ord=2)
    dt = np.pi / norm
    return dt, norm

dt, spec_norm = get_kqd_dt(H_op, N)
print(f"dt = {dt:.6f}, spectral norm = {spec_norm:.4f}")

# ============================================================
# Part 1: Trotter step 수에 따른 KQD 에너지 수렴 + 회로 메트릭
# ============================================================
print("\n" + "=" * 80)
print("Part 1: Trotter step 수에 따른 KQD 에너지 및 회로 메트릭 비교")
print("=" * 80)

import itertools as it

trotter_steps_list = [1, 2, 4, 6]
r_dim = 5

aer_backend = AerSimulator()
is_eigen, E_vac = is_vacuum_eigenstate(H_op)
print(f"  Vacuum eigenstate check: {is_eigen}, E_vac = {E_vac.real:.4f}")

state_prep = get_neel_state(N)
neel_sv = Statevector(state_prep)
H_expval_t0 = neel_sv.expectation_value(H_op)

all_kqd_results = {}

for ts in trotter_steps_list:
    print(f"\n--- Trotter Steps = {ts} ---")
    dt_circ = dt / ts
    print(f"  dt_circ = {dt_circ:.6f}")
    
    t_param = Parameter("t")
    qc_template = build_kqd_template_circuit(
        n_qubits=N,
        state_prep_circuit=state_prep,
        H_op=H_op,
        t_param=t_param,
        num_trotter_steps=ts,
        synthesis_name="lie"
    )
    
    qc_trans = transpile(qc_template, backend=aer_backend, optimization_level=2)
    depth = qc_trans.depth()
    two_q = qc_trans.num_nonlocal_gates()
    print(f"  회로 깊이: {depth}, 2Q 게이트: {two_q}")
    
    # Observables 생성
    obs_S_real = SparsePauliOp("I" * N).tensor(SparsePauliOp("X"))
    obs_S_imag = SparsePauliOp("I" * N).tensor(SparsePauliOp("Y"))
    observables_list = [[obs_S_real], [obs_S_imag]]
    for p, coeff in zip(H_op.paulis, H_op.coeffs):
        obs_H_real = SparsePauliOp(p).tensor(SparsePauliOp("X"))
        obs_H_imag = SparsePauliOp(p).tensor(SparsePauliOp("Y"))
        observables_list.append([obs_H_real])
        observables_list.append([obs_H_imag])
    
    if qc_trans.layout is not None:
        observables_physical = [
            [obs[0].apply_layout(qc_trans.layout, num_qubits=aer_backend.num_qubits)]
            for obs in observables_list
        ]
    else:
        observables_physical = observables_list
    
    # Parameter sweep
    parameters_val = [dt_circ * k for k in range(1, r_dim)]
    params_sweep = np.vstack(parameters_val).T
    
    # Run
    estimator = get_estimator("noiseless")
    pub = (qc_trans, observables_physical, params_sweep)
    job = estimator.run([pub], precision=0.01)
    results = job.result()[0]
    
    # Prefactors
    prefactors = [np.exp(-1j * E_vac * k * dt_circ) for k in range(1, r_dim)]
    
    # S matrix
    S_first_row = np.zeros(r_dim, dtype=complex)
    S_first_row[0] = 1.0 + 0j
    for k in range(r_dim - 1):
        expval_real = float(results.data.evs[0][k])
        expval_imag = float(results.data.evs[1][k])
        S_first_row[k + 1] = prefactors[k] * (expval_real + 1j * expval_imag)
    
    S_matrix = np.zeros((r_dim, r_dim), dtype=complex)
    for i, j in it.product(range(r_dim), repeat=2):
        if i >= j:
            S_matrix[j, i] = S_first_row[i - j]
        else:
            S_matrix[j, i] = np.conj(S_first_row[j - i])
    
    # H matrix
    H_first_row = np.zeros(r_dim, dtype=complex)
    H_first_row[0] = H_expval_t0
    for obs_idx, coeff in enumerate(H_op.coeffs):
        for k in range(r_dim - 1):
            expval_real = float(results.data.evs[2 + 2 * obs_idx][k])
            expval_imag = float(results.data.evs[2 + 2 * obs_idx + 1][k])
            H_first_row[k + 1] += prefactors[k] * coeff * (expval_real + 1j * expval_imag)
    
    H_matrix = np.zeros((r_dim, r_dim), dtype=complex)
    for i, j in it.product(range(r_dim), repeat=2):
        if i >= j:
            H_matrix[j, i] = H_first_row[i - j]
        else:
            H_matrix[j, i] = np.conj(H_first_row[j - i])
    
    # Solve generalized eigenvalue problem for each d
    gnd_energies = []
    for d in range(1, r_dim + 1):
        energy, num_good = solve_regularized_gen_eig(
            H_matrix[:d, :d], S_matrix[:d, :d], threshold=1e-2, return_dimn=True
        )
        gnd_energies.append(energy)
        print(f"    d={d} | Good Vecs: {num_good} | E = {energy:.6f}")
    
    all_kqd_results[ts] = {
        "trotter_steps": ts,
        "dt_circ": dt_circ,
        "depth": depth,
        "two_q_gates": two_q,
        "kqd_energies": gnd_energies,
    }

# Save to JSON
with open('results/kqd_trotter_sweep.json', 'w') as f:
    json.dump(all_kqd_results, f, indent=4)

# --- Plot 1a: KQD Energy vs Krylov Dimension for different Trotter steps ---
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
markers = ['s', 'D', '^', 'o']
for idx, ts in enumerate(trotter_steps_list):
    data = all_kqd_results[ts]
    label = f"Trotter {ts}-step (Depth={data['depth']}, 2Q={data['two_q_gates']})"
    ax.plot(range(1, r_dim + 1), data['kqd_energies'], 
            color=colors[idx], marker=markers[idx], markersize=8, linewidth=2,
            linestyle='--', label=label)
ax.axhline(y=exact_gnd, color='black', linestyle='-', linewidth=2, label=f"Exact GS = {exact_gnd:.4f}")
ax.set_xlabel("Krylov Dimension ($d$)", fontsize=13)
ax.set_ylabel("Ground State Energy", fontsize=13)
ax.set_title("KQD Energy Convergence vs Trotter Steps (Noiseless)", fontsize=14, fontweight='bold')
ax.set_xticks(range(1, r_dim + 1))
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/kqd_trotter_sweep_energy.png', dpi=150)
plt.close()
print("\nPlot saved: results/kqd_trotter_sweep_energy.png")

# --- Plot 1b: Circuit Depth & 2Q Gates vs Trotter Steps ---
fig, ax1 = plt.subplots(figsize=(8, 5))
depths = [all_kqd_results[ts]['depth'] for ts in trotter_steps_list]
gates = [all_kqd_results[ts]['two_q_gates'] for ts in trotter_steps_list]

color1 = '#3498db'
color2 = '#e74c3c'
ax1.bar([x - 0.2 for x in range(len(trotter_steps_list))], depths, 0.4, 
        label='Circuit Depth', color=color1, alpha=0.85)
ax2 = ax1.twinx()
ax2.bar([x + 0.2 for x in range(len(trotter_steps_list))], gates, 0.4, 
        label='2Q Gates', color=color2, alpha=0.85)

ax1.set_xlabel("Trotter Steps", fontsize=13)
ax1.set_ylabel("Circuit Depth", fontsize=13, color=color1)
ax2.set_ylabel("2-Qubit Gate Count", fontsize=13, color=color2)
ax1.set_xticks(range(len(trotter_steps_list)))
ax1.set_xticklabels([str(ts) for ts in trotter_steps_list])
ax1.set_title("KQD Circuit Complexity vs Trotter Steps", fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
ax1.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/kqd_circuit_metrics.png', dpi=150)
plt.close()
print("Plot saved: results/kqd_circuit_metrics.png")


# ============================================================
# Part 2: 표준 Trotterization FFT 분석
# ============================================================
print("\n" + "=" * 80)
print("Part 2: 표준 Trotterization 데이터 FFT 분석 (에너지 추출)")
print("=" * 80)

# Load noiseless data
with open('results/std_xxz_noiseless.json') as f:
    std_noiseless = json.load(f)
with open('results/std_xxz_fake.json') as f:
    std_fake = json.load(f)

def fft_energy_analysis(times, expectations, label, dt_sample):
    """
    고전적 FFT를 사용하여 시간 동역학 데이터에서 에너지 주파수 성분을 추출합니다.
    <Z_0(t)> = sum_k c_k * exp(-i * omega_k * t) 에서 omega_k를 추출합니다.
    """
    signal = np.array(expectations)
    N_pts = len(signal)
    
    # FFT 수행
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N_pts, d=dt_sample)
    
    # 양의 주파수만 선택
    pos_mask = fft_freqs >= 0
    fft_power = np.abs(fft_vals[pos_mask]) ** 2 / N_pts
    freqs_pos = fft_freqs[pos_mask]
    
    # 각진동수 omega = 2*pi*freq
    omegas = 2 * np.pi * freqs_pos
    
    # 피크 검출 (파워가 상위 30% 이상)
    threshold = np.max(fft_power) * 0.05
    peak_indices = np.where(fft_power > threshold)[0]
    peak_omegas = omegas[peak_indices]
    peak_powers = fft_power[peak_indices]
    
    print(f"\n  [{label}] FFT Analysis:")
    print(f"    Number of data points: {N_pts}")
    print(f"    dt_sample: {dt_sample:.4f}")
    print(f"    Peak frequencies (omega):")
    for idx in peak_indices:
        print(f"      omega = {omegas[idx]:.4f}, Power = {fft_power[idx]:.6f}")
    
    return freqs_pos, fft_power, omegas, peak_indices

dt_sample = 0.1  # 시간 간격

freqs_n, power_n, omegas_n, peaks_n = fft_energy_analysis(
    std_noiseless['times'], std_noiseless['expectations'], 
    "Noiseless", dt_sample
)
freqs_f, power_f, omegas_f, peaks_f = fft_energy_analysis(
    std_fake['times'], std_fake['expectations'], 
    "Fake (Noisy)", dt_sample
)

# --- Plot 2a: Time-domain comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(std_noiseless['times'], std_noiseless['expectations'], 'o-', 
        color='#3498db', linewidth=2, markersize=5, label='Noiseless')
ax.plot(std_fake['times'], std_fake['expectations'], 's-', 
        color='#e74c3c', linewidth=2, markersize=5, label='Noisy (FakeBrisbane)')
ax.set_xlabel("Time $t$", fontsize=13)
ax.set_ylabel(r"$\langle Z_0(t) \rangle$", fontsize=13)
ax.set_title("Standard Trotterization: Spin Dynamics", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(omegas_n, power_n, '-', color='#3498db', linewidth=2, label='Noiseless')
ax.plot(omegas_f, power_f, '-', color='#e74c3c', linewidth=2, alpha=0.7, label='Noisy (FakeBrisbane)')

# 정확한 에너지 차이 마킹 (Neel 초기상태의 에너지가 E_neel일 때, 천이 주파수는 |E_k - E_l|)
# Z_0(t) = sum_{k,l} |c_k|^2 * <k|Z_0|l> * exp(-i(E_k - E_l)t)
# 따라서 FFT 피크는 에너지 차이(transition frequency)에 해당
# Neel 상태의 고유에너지 분해로부터 계산
H_mat = H_op.to_matrix()
neel_vec = neel_sv.data
eigvals, eigvecs = np.linalg.eigh(H_mat)
# Neel 상태를 고유상태로 분해
coeffs = eigvecs.T @ neel_vec  # 고유상태 계수
Z0_op = SparsePauliOp("IIIZ").to_matrix()

# 유의미한 천이 주파수 계산
transition_freqs = []
for k in range(len(eigvals)):
    for l in range(k+1, len(eigvals)):
        ck = coeffs[k]
        cl = coeffs[l]
        matrix_elem = eigvecs[:, k].conj() @ Z0_op @ eigvecs[:, l]
        weight = np.abs(ck * cl.conj() * matrix_elem)
        if weight > 0.01:
            omega_kl = np.abs(eigvals[l] - eigvals[k])
            transition_freqs.append((omega_kl, weight))
            print(f"  Transition: E_{l} - E_{k} = {eigvals[l]:.4f} - {eigvals[k]:.4f} = {omega_kl:.4f}, weight = {weight:.4f}")

for omega_t, w in transition_freqs:
    ax.axvline(x=omega_t, color='green', linestyle='--', alpha=0.6, linewidth=1)
ax.axvline(x=0, color='green', linestyle='--', alpha=0.6, linewidth=1, label='Exact transitions')

ax.set_xlabel(r"$\omega$ (angular frequency)", fontsize=13)
ax.set_ylabel("FFT Power", fontsize=13)
ax.set_title("FFT Spectral Analysis", fontsize=13, fontweight='bold')
ax.set_xlim(-1, 35)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/std_fft_analysis.png', dpi=150)
plt.close()
print("\nPlot saved: results/std_fft_analysis.png")


# ============================================================
# Part 3: 표준 Trotterization 회로 메트릭 수집
# ============================================================
print("\n" + "=" * 80)
print("Part 3: 표준 Trotterization 회로 메트릭 수집")
print("=" * 80)

fake_backend = FakeBrisbane()
std_metrics = {}

# 각 시간 포인트에서 최대 깊이를 기록
for backend_name, backend_inst in [("AerSimulator", aer_backend), ("FakeBrisbane", fake_backend)]:
    # 마지막 시점 (t=2.0)의 회로 메트릭
    t_final = 2.0
    steps_final = 10
    qc_std = get_neel_state(N).compose(
        build_trotter_circuit(H_op, time=t_final, steps=steps_final, model_name="suzuki_2")
    )
    qc_std_trans = transpile(qc_std, backend=backend_inst, optimization_level=2)
    depth = qc_std_trans.depth()
    two_q = qc_std_trans.num_nonlocal_gates()
    print(f"\n  [{backend_name}] t=2.0, steps=10, suzuki_2:")
    print(f"    Depth: {depth}, 2Q Gates: {two_q}")
    
    # 초기 시점 (t=0.1, steps=1)의 회로 메트릭
    t_init = 0.1
    steps_init = 1
    qc_init = get_neel_state(N).compose(
        build_trotter_circuit(H_op, time=t_init, steps=steps_init, model_name="suzuki_2")
    )
    qc_init_trans = transpile(qc_init, backend=backend_inst, optimization_level=2)
    depth_init = qc_init_trans.depth()
    two_q_init = qc_init_trans.num_nonlocal_gates()
    print(f"  [{backend_name}] t=0.1, steps=1, suzuki_2:")
    print(f"    Depth: {depth_init}, 2Q Gates: {two_q_init}")
    
    std_metrics[backend_name] = {
        "t_2.0_depth": depth,
        "t_2.0_2q_gates": two_q,
        "t_0.1_depth": depth_init,
        "t_0.1_2q_gates": two_q_init,
    }

with open('results/std_circuit_metrics.json', 'w') as f:
    json.dump(std_metrics, f, indent=4)
print("\nMetrics saved: results/std_circuit_metrics.json")

# ============================================================
# 최종 KQD 수렴 그래프 업데이트 (2-step 기본 설정으로)
# ============================================================
print("\n" + "=" * 80)
print("Part 4: KQD 수렴 그래프 업데이트 (6-step, noiseless)")
print("=" * 80)

# 6-step noiseless 결과 사용
best_result = all_kqd_results[6]
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, r_dim + 1), best_result['kqd_energies'], 'bo--', 
        markersize=9, linewidth=2, label="KQD Estimate (6-step Trotter)")
ax.axhline(y=exact_gnd, color='r', linestyle='-', linewidth=2, 
           label=f"Exact GS = {exact_gnd:.4f}")
ax.set_xticks(range(1, r_dim + 1))
ax.set_xlabel("Krylov Dimension ($d$)", fontsize=13)
ax.set_ylabel("Energy", fontsize=13)
ax.set_title("KQD Ground State Energy Convergence (Noiseless, 6-step)", 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/kqd_convergence_noiseless.png', dpi=150)
plt.close()
print("Updated: results/kqd_convergence_noiseless.png")

print("\n모든 분석이 완료되었습니다!")
