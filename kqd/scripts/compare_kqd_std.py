import os
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian
from src.circuit_builder import build_trotter_circuit
from src.kqd import build_kqd_template_circuit

def get_neel_state(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        qc.x(i)
    return qc

def main():
    N = 4
    # XXZ Hamiltonian
    H_op = get_xyz_hamiltonian(N, 1.0, 1.0, 0.5, 0.5)
    
    # Backends
    aer = AerSimulator()
    fake = FakeBrisbane()
    
    print("=" * 80)
    print("기존 Trotterization vs KQD 회로 메트릭 비교 (XXZ 해밀토니안)")
    print("=" * 80)
    
    # 1. 기존 Trotterization 회로 (t=2.0, steps=10, suzuki_2)
    # run_harness에서 최종 시점 t=2.0을 분석하기 위해 사용되는 깊은 회로입니다.
    qc_std_base = get_neel_state(N).compose(
        build_trotter_circuit(H_op, time=2.0, steps=10, model_name="suzuki_2")
    )
    qc_std_aer = transpile(qc_std_base, backend=aer, optimization_level=2)
    qc_std_fake = transpile(qc_std_base, backend=fake, optimization_level=2)
    
    # 2. KQD 템플릿 회로 (r=5, steps=2, lie)
    from qiskit.circuit import Parameter
    t_param = Parameter("t")
    qc_kqd_base = build_kqd_template_circuit(
        n_qubits=N,
        state_prep_circuit=get_neel_state(N),
        H_op=H_op,
        t_param=t_param,
        num_trotter_steps=2,
        synthesis_name="lie"
    )
    qc_kqd_aer = transpile(qc_kqd_base, backend=aer, optimization_level=2)
    qc_kqd_fake = transpile(qc_kqd_base, backend=fake, optimization_level=2)
    
    print("\n[1] Noiseless Simulator (AerSimulator)")
    print(f"  - 기존 Trotter (t=2.0, steps=10, suzuki_2) | 큐비트 수: {qc_std_aer.num_qubits} | 회로 깊이: {qc_std_aer.depth():<4} | 2Q 게이트: {qc_std_aer.num_nonlocal_gates()}")
    print(f"  - KQD 템플릿  (r=5, steps=2, lie)         | 큐비트 수: {qc_kqd_aer.num_qubits} | 회로 깊이: {qc_kqd_aer.depth():<4} | 2Q 게이트: {qc_kqd_aer.num_nonlocal_gates()}")
    
    print("\n[2] Noisy Device Simulator (FakeBrisbane)")
    print(f"  - 기존 Trotter (t=2.0, steps=10, suzuki_2) | 큐비트 수: {qc_std_fake.num_qubits} | 회로 깊이: {qc_std_fake.depth():<4} | 2Q 게이트: {qc_std_fake.num_nonlocal_gates()}")
    print(f"  - KQD 템플릿  (r=5, steps=2, lie)         | 큐비트 수: {qc_kqd_fake.num_qubits} | 회로 깊이: {qc_kqd_fake.depth():<4} | 2Q 게이트: {qc_kqd_fake.num_nonlocal_gates()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
