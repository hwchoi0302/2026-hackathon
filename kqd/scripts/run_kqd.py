import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import itertools as it

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian
from src.mitigation import configure_adaptive_mitigation
from src.execution import get_estimator
from src.kqd import (
    solve_regularized_gen_eig,
    is_vacuum_eigenstate,
    build_kqd_template_circuit
)

# KQD dt 계산 함수 (메뉴얼 수식 기준)
def get_kqd_dt(H_op, n_qubits):
    single_particle_H = np.zeros((n_qubits, n_qubits), dtype=complex)
    for i in range(n_qubits):
        for j in range(i + 1):
            for p, coeff in H_op.to_list():
                p_x = p.count('X') # or Pauli(p).x
                # Qiskit SparsePauliOp의 label에서 x, z 파트 추출
                # label 문자열에서 각 비트의 X, Z 여부를 판단
                # little-endian이므로 뒤집어서 검사
                p_rev = p[::-1]
                x_bits = [k for k, char in enumerate(p_rev) if char in ('X', 'Y')]
                z_bits = [k for k, char in enumerate(p_rev) if char in ('Z', 'Y')]
                
                # 단일 입자 상태 |i>와 |j> 사이의 해밀토니안 행렬 요소를 구합니다.
                # |i>는 i번째 큐비트만 1인 상태이고, |j>는 j번째 큐비트만 1인 상태입니다.
                # 두 상태의 x 비트 차이가 p_rev의 X 비트 세트와 같은지 확인
                diff_set = set([i, j])
                if set(x_bits) == diff_set or (i == j and len(x_bits) == 0):
                    # 부호 계산
                    num_y = sum(1 for char in p_rev if char == 'Y')
                    # sgn = (-1j)**(Y 개수) * (-1)**(Z가 걸린 1인 비트들)
                    # 여기서는 간단히 Z Pauli 연산의 기댓값을 구합니다.
                    # Z 연산자가 j번째 비트에 있으면 sign은 -1이 됩니다.
                    z_sign = 1
                    for z_bit in z_bits:
                        if z_bit == i: # state |i> has 1 at position i
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

def get_neel_state(num_qubits: int) -> QuantumCircuit:
    """Neel 상태 |1010...> 준비"""
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        qc.x(i)
    return qc

def main():
    parser = argparse.ArgumentParser(description="KQD Trotterization Simulation")
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["noiseless", "fake", "real"], 
        default="noiseless",
        help="실행할 백엔드: noiseless, fake (FakeBrisbane), 또는 real (ibm_yonsei)."
    )
    parser.add_argument(
        "--enable-dd", 
        action="store_true", 
        default=True,
        help="동적 디커플링(DD) 활성화 여부"
    )
    parser.add_argument(
        "--dd-sequence", 
        type=str, 
        default="XY4", 
        choices=["XX", "XY4"],
        help="사용할 DD 시퀀스"
    )
    parser.add_argument(
        "--enable-trex", 
        action="store_true", 
        default=True,
        help="Twirled Readout Mitigation (TREX) 활성화 여부"
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=4,
        help="큐비트 수 (기본값: 4)"
    )
    parser.add_argument(
        "--r-dim",
        type=int,
        default=5,
        help="Krylov 차원 r (기본값: 5)"
    )
    parser.add_argument(
        "--trotter-steps",
        type=int,
        default=6,
        help="Trotter 스텝 수 (기본값: 6)"
    )
    parser.add_argument(
        "--depth-threshold",
        type=int,
        default=80,
        help="ZNE 활성화를 위한 최대 회로 깊이 임계값"
    )
    parser.add_argument(
        "--gate-threshold",
        type=int,
        default=100,
        help="ZNE 활성화를 위한 최대 2-큐비트 게이트 수 임계값"
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.01,
        help="Estimator 실행 정밀도 (기본값: 0.01)"
    )
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    print("=" * 80)
    print(f"KQD Trotterization 시뮬레이션 시작 (백엔드: {args.backend.upper()})")
    print("=" * 80)

    # 1. 백엔드 설정
    backend_instance = None
    if args.backend == "noiseless":
        backend_instance = AerSimulator()
        print("  [Simulator] Noiseless AerSimulator 사용")
    elif args.backend == "fake":
        backend_instance = FakeBrisbane()
        print(f"  [Simulator] Noisy FakeBrisbane ({backend_instance.num_qubits} qubits) 사용")
    elif args.backend == "real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        iam_key = os.environ.get("IBM_IAM_APIKEY", "")
        crn = os.environ.get("IBM_CRN", "")
        if iam_key and crn:
            service = QiskitRuntimeService(channel="ibm_cloud", token=iam_key, instance=crn)
        else:
            print("  IBM Quantum 계정 로드 중...")
            service = QiskitRuntimeService()
        backend_instance = service.backend("ibm_yonsei")
        print(f"  [QPU] Real 하드웨어 연결됨: {backend_instance.name}")

    # 2. XXZ 해밀토니안 구성 (대칭성 확보를 위해 Jx=Jy=1.0 설정)
    N = args.num_qubits
    JX, JY, JZ, h_val = 1.0, 1.0, 0.5, 0.5
    H_op = get_xyz_hamiltonian(N, JX, JY, JZ, h_val)
    print(f"  XXZ 해밀토니안 구성 완료 (N={N}, Jx={JX}, Jy={JY}, Jz={JZ}, h={h_val})")
    
    is_eigen, E_vac = is_vacuum_eigenstate(H_op)
    print(f"  진공 상태 |0>^N가 고유 상태인가? {is_eigen} (고유값: {E_vac.real:.4f})")
    if not is_eigen:
        print("  [경고] 진공 상태가 고유 상태가 아닙니다! Efficient Hadamard Test가 부정확할 수 있습니다.")

    # 3. KQD dt 계산 (매뉴얼 수식 기반)
    dt, spec_norm = get_kqd_dt(H_op, N)
    print(f"  단일 입자 해밀토니안 spectral norm: {spec_norm:.4f}")
    print(f"  계산된 최적 dt: {dt:.6f}")
    
    dt_circ = dt / args.trotter_steps
    print(f"  단일 Trotter step 시간 간격 (dt_circ): {dt_circ:.6f}")
    if dt_circ > 0.20:
        print(f"  [경고] 단일 Trotter step 시간 간격 ({dt_circ:.6f})이 너무 큽니다 (> 0.2).")
        print(f"         Trotter 오차로 인해 [H, U_Trotter] = 0 가정이 훼손되어,")
        print(f"         KQD의 generalized eigenvalue가 exact ground state 이하로 발산할 수 있습니다 (변분 하한 위배).")
        print(f"         안정적인 수렴을 위해 --trotter-steps를 더 높게(예: 4 또는 6 이상) 설정할 것을 권장합니다.")

    # 4. 초기 상태 설정 (Neel State |1010...>)
    state_prep = get_neel_state(N)
    print(f"  초기 상태 (Neel State) 회로 준비 완료")

    # 5. KQD 템플릿 회로 구성
    t_param = Parameter("t")
    qc_template = build_kqd_template_circuit(
        n_qubits=N,
        state_prep_circuit=state_prep,
        H_op=H_op,
        t_param=t_param,
        num_trotter_steps=args.trotter_steps,
        synthesis_name="lie"
    )

    # 6. 트랜스파일을 통한 회로 분석 및 적응형 에러 완화 설정
    print("  템플릿 회로 가상 트랜스파일 중...")
    qc_trans = transpile(qc_template, backend=backend_instance, optimization_level=2)
    depth = qc_trans.depth()
    two_q_gates = qc_trans.num_nonlocal_gates()
    print(f"  [트랜스파일 결과] 회로 깊이: {depth} | 2-큐비트 게이트 수: {two_q_gates}")

    # 에러 완화 옵션 결정
    # Aer의 경우 EstimatorOptions 객체가 없으므로 딕셔너리로 초기화
    if args.backend in ["noiseless", "fake"]:
        options = {}
    else:
        from qiskit_ibm_runtime.options import EstimatorOptions
        options = EstimatorOptions()

    options, zne_enabled = configure_adaptive_mitigation(
        options,
        depth=depth,
        two_q_gates=two_q_gates,
        depth_threshold=args.depth_threshold,
        two_q_gate_threshold=args.gate_threshold,
        enable_dd=args.enable_dd,
        dd_sequence=args.dd_sequence,
        enable_trex=args.enable_trex
    )
    print(f"  [적응형 에러 완화] ZNE 활성화 여부: {zne_enabled} (DD: {args.enable_dd}, TREX: {args.enable_trex})")

    # 7. Estimator 인스턴스 획득
    # AerEstimator에 딕셔너리 옵션을 전달하는 방식 호환
    if args.backend == "noiseless":
        estimator = get_estimator("noiseless")
    elif args.backend == "fake":
        estimator = get_estimator("fake", backend_instance=backend_instance)
    elif args.backend == "real":
        estimator = get_estimator("real", backend_instance=backend_instance, 
                                  enable_dd=args.enable_dd, dd_sequence=args.dd_sequence, 
                                  enable_trex=args.enable_trex)
        # ZNE 옵션을 수동으로 Estimator에 업데이트 (configure_adaptive_mitigation가 수정한 options 객체 이용)
        estimator.options.update(options)

    # 8. 관측량 (Observables) 빌드 및 레이아웃 적용
    # S 관측량
    obs_S_real = SparsePauliOp("I" * N).tensor(SparsePauliOp("X"))
    obs_S_imag = SparsePauliOp("I" * N).tensor(SparsePauliOp("Y"))
    
    # H 관측량
    observables_list = [[obs_S_real], [obs_S_imag]]
    for p, coeff in zip(H_op.paulis, H_op.coeffs):
        obs_H_real = SparsePauliOp(p).tensor(SparsePauliOp("X"))
        obs_H_imag = SparsePauliOp(p).tensor(SparsePauliOp("Y"))
        observables_list.append([obs_H_real])
        observables_list.append([obs_H_imag])

    # 레이아웃 적용
    if qc_trans.layout is not None:
        print("  물리적 큐비트 레이아웃을 관측량에 적용 중...")
        observables_physical = [
            [obs[0].apply_layout(qc_trans.layout, num_qubits=backend_instance.num_qubits)]
            for obs in observables_list
        ]
    else:
        observables_physical = observables_list

    # 9. 파라미터 스위핑 설정
    dt_circ = dt / args.trotter_steps
    parameters_val = [dt_circ * k for k in range(1, args.r_dim)]
    # Qiskit V2 sweep shape: (1, r-1)
    params_sweep = np.vstack(parameters_val).T
    print(f"  시간 파라미터 스위프 설정: {parameters_val}")

    # 10. 회로 실행
    pub = (qc_trans, observables_physical, params_sweep)
    print("  양자 시뮬레이션 job 제출 중...")
    job = estimator.run([pub], precision=args.precision)
    if hasattr(job, "job_id"):
        print(f"  제출된 Job ID: {job.job_id()}")
    
    results = job.result()[0]
    print("  시뮬레이션 완료 및 결과 획득")

    # 11. 결과 포스트 프로세싱 및 행렬 복원
    # 진공 위상 팩터 계산
    prefactors = [
        np.exp(-1j * E_vac * k * dt_circ)
        for k in range(1, args.r_dim)
    ]

    # S (Overlap) 행렬 복원
    S_first_row = np.zeros(args.r_dim, dtype=complex)
    S_first_row[0] = 1.0 + 0j
    for k in range(args.r_dim - 1):
        # Qiskit V2 results.data.evs shape is (num_observables, num_sweep_points)
        expval_real = float(results.data.evs[0][k])
        expval_imag = float(results.data.evs[1][k])
        S_first_row[k + 1] = prefactors[k] * (expval_real + 1j * expval_imag)

    print("  [DEBUG] S_first_row:", S_first_row)

    S_matrix = np.zeros((args.r_dim, args.r_dim), dtype=complex)
    for i, j in it.product(range(args.r_dim), repeat=2):
        if i >= j:
            S_matrix[j, i] = S_first_row[i - j]
        else:
            S_matrix[j, i] = np.conj(S_first_row[j - i])

    # H (Effective Hamiltonian) 행렬 복원
    # t=0에서의 H 기댓값 계산 (Neel State에서 해밀토니안의 기대값)
    neel_statevector = Statevector(state_prep)
    H_expval_t0 = neel_statevector.expectation_value(H_op)

    H_first_row = np.zeros(args.r_dim, dtype=complex)
    H_first_row[0] = H_expval_t0

    for obs_idx, coeff in enumerate(H_op.coeffs):
        for k in range(args.r_dim - 1):
            expval_real = float(results.data.evs[2 + 2 * obs_idx][k])
            expval_imag = float(results.data.evs[2 + 2 * obs_idx + 1][k])
            H_first_row[k + 1] += prefactors[k] * coeff * (expval_real + 1j * expval_imag)

    print("  [DEBUG] H_first_row:", H_first_row)

    H_matrix = np.zeros((args.r_dim, args.r_dim), dtype=complex)
    for i, j in it.product(range(args.r_dim), repeat=2):
        if i >= j:
            H_matrix[j, i] = H_first_row[i - j]
        else:
            H_matrix[j, i] = np.conj(H_first_row[j - i])

    # 12. 일반화 고유값 문제 풀기 및 고전 엄밀해 계산
    # KQD 에너지 예측 (차원 d = 1 ~ r)
    gnd_energies = []
    print("\n  [KQD 에너지 계산 결과]")
    for d in range(1, args.r_dim + 1):
        # Debug print
        s_vals = np.linalg.eigvalsh(S_matrix[:d, :d])
        energy, num_good = solve_regularized_gen_eig(
            H_matrix[:d, :d], S_matrix[:d, :d], threshold=1e-2, return_dimn=True
        )
        gnd_energies.append(energy)
        print(f"    Krylov 차원 d = {d} | S 고유값: {s_vals} | Good Vecs: {num_good} | Ground State 에너지 예측값: {energy:.6f}")

    # 고전적 엄밀해 계산 (Full Diagonalization)
    exact_eigvals = np.linalg.eigvalsh(H_op.to_matrix())
    exact_gnd = exact_eigvals[0]
    print(f"\n  고전 엄밀해 Ground State 에너지: {exact_gnd:.6f}")

    # 13. 시각화 및 결과 저장
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.r_dim + 1), gnd_energies, 'bo--', label="KQD Estimate")
    plt.axhline(y=exact_gnd, color='r', linestyle='-', label="Exact Ground State")
    plt.xticks(range(1, args.r_dim + 1))
    plt.xlabel("Krylov Dimension ($d$)", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    plt.title(f"KQD Ground State Energy Convergence ({args.backend.upper()})", fontsize=14, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plot_path = f"results/kqd_convergence_{args.backend}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  에너지 수렴 그래프 저장 완료: {plot_path}")

    # 데이터 저장
    result_data = {
        "backend": args.backend,
        "num_qubits": N,
        "r_dim": args.r_dim,
        "dt": dt,
        "spec_norm": spec_norm,
        "depth": depth,
        "two_q_gates": two_q_gates,
        "zne_enabled": zne_enabled,
        "kqd_energies": gnd_energies,
        "exact_gnd_energy": exact_gnd
    }
    json_path = f"results/kqd_results_{args.backend}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"  결과 데이터 JSON 저장 완료: {json_path}")
    print("\nKQD 실행이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()
