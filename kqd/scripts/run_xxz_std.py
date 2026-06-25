import os
import sys
import argparse
import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian
from src.circuit_builder import build_trotter_circuit
from src.execution import get_estimator

def get_neel_state(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        qc.x(i)
    return qc

def main():
    parser = argparse.ArgumentParser(description="Standard Trotterization Simulation for XXZ model")
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["noiseless", "fake"], 
        default="noiseless",
        help="Backend: noiseless or fake."
    )
    args = parser.parse_args()

    N = 4
    JX, JY, JZ, h = 1.0, 1.0, 0.5, 0.5
    H_op = get_xyz_hamiltonian(N, JX, JY, JZ, h)
    observable = SparsePauliOp("IIIZ")
    
    times = np.arange(0.0, 2.1, 0.1)
    expectations = []
    
    # Setup backend
    if args.backend == "noiseless":
        backend_instance = AerSimulator()
        estimator = get_estimator("noiseless")
    else:
        backend_instance = FakeBrisbane()
        estimator = get_estimator("fake", backend_instance=backend_instance)
        
    print(f"Standard Trotterization 시뮬레이션 시작 (백엔드: {args.backend.upper()})")
    
    # 1. t=0.0 기댓값 계산
    neel_sv = Statevector(get_neel_state(N))
    expectations.append(float(neel_sv.expectation_value(observable).real))
    print(f"  t=0.00 | <Z_0>: {expectations[0]:.4f} (고전 계산)")
    
    # 2. t > 0.0 기댓값 계산을 위한 양자 회로 일괄 제출
    pubs = []
    active_times = []
    for t in times:
        if t == 0.0:
            continue
        active_times.append(t)
        steps = max(1, int(np.round(t / 0.2)))
        
        qc_trotter = build_trotter_circuit(H_op, time=t, steps=steps, model_name="suzuki_2")
        qc = get_neel_state(N).compose(qc_trotter)
        
        qc_exec = transpile(qc, backend=backend_instance, optimization_level=2)
        
        if qc_exec.layout is not None:
            obs_physical = observable.apply_layout(qc_exec.layout, num_qubits=backend_instance.num_qubits)
        else:
            obs_physical = observable
            
        pubs.append((qc_exec, obs_physical))
        
    job = estimator.run(pubs, precision=0.01)
    results = job.result()
    
    for idx, t in enumerate(active_times):
        val = float(results[idx].data.evs)
        expectations.append(val)
        print(f"  t={t:.2f} (r={max(1, int(np.round(t/0.2)))}) | <Z_0>: {val:.4f}")
        
    # 결과 저장
    result_data = {
        "times": list(times),
        "expectations": expectations
    }
    os.makedirs('results', exist_ok=True)
    json_path = f"results/std_xxz_{args.backend}.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"성공적으로 결과를 저장했습니다: {json_path}")

if __name__ == "__main__":
    main()
