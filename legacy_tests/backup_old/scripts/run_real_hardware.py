"""
Real IBM Quantum hardware execution script.

IBM Quantum Network 백엔드(ibm_yonsei)에 접근하는 두 가지 방법:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
방법 A: IBM Cloud IAM API Key + CRN (현재 보유한 CRN 사용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IAM API Key 발급 방법:
  1. https://cloud.ibm.com 로그인
  2. 상단 메뉴 Manage -> Access (IAM) -> API keys
  3. [Create an IBM Cloud API key] 클릭 -> 이름 입력 -> Create
  4. 생성된 키를 복사 (abc123... 형태의 44자 문자열)
     ※ IBMid-66300525OM 은 로그인 ID이지 API 키가 아닙니다.

실행:
  export IBM_IAM_APIKEY="<발급받은 IAM API Key>"
  export IBM_CRN="crn:v~"
  python scripts/run_real_hardware.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
방법 B: IBM Quantum Platform Token (ibm_quantum 채널)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. https://quantum.ibm.com 로그인
  2. 우측 상단 이름 -> Copy API token
  3. export IBM_QUANTUM_TOKEN="<복사한 토큰>"
     export USE_QUANTUM_CHANNEL=1
  python scripts/run_real_hardware.py
"""

import os
import sys
import json
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile, QuantumCircuit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.heisenberg import build_second_order_trotter
from src.error_mitigation import apply_zne

# ── Parameters ─────────────────────────────────────────────────────────────────
N = 4
J = 1.0
h = 0.5
T = 1.0
ZNE_R = 5
SHOTS = 8192
BACKEND_NAME = "ibm_yonsei"


def connect_ibm_cloud(iam_key: str, crn: str) -> QiskitRuntimeService:
    """IBM Cloud 채널로 연결 (IAM API Key + CRN)"""
    print("[ibm_cloud channel] Saving account...")
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",
        token=iam_key,
        instance=crn,
        overwrite=True,
        set_as_default=True,
    )
    service = QiskitRuntimeService(channel="ibm_cloud", instance=crn)
    backends = [b.name for b in service.backends()]
    print(f"Connected. Available backends: {backends}")
    if BACKEND_NAME not in backends:
        print(f"WARNING: '{BACKEND_NAME}' not found in your cloud instance.")
    return service


def connect_ibm_quantum(token: str) -> QiskitRuntimeService:
    """IBM Quantum Platform 채널로 연결 (ibm_quantum token)"""
    print("[ibm_quantum channel] Saving account...")
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        overwrite=True,
        set_as_default=True,
    )
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backends = [b.name for b in service.backends()]
    print(f"Connected. Available backends: {backends}")
    if BACKEND_NAME not in backends:
        print(f"WARNING: '{BACKEND_NAME}' not found. Check your access.")
    return service


def connect_from_saved() -> QiskitRuntimeService:
    """이전에 save_account로 저장된 자격증명으로 연결 시도"""
    print("[Saved credentials] Attempting to load saved account...")
    service = QiskitRuntimeService()   # loads default saved account
    backends = [b.name for b in service.backends()]
    print(f"Connected. Available backends: {backends}")
    return service


def run_on_real_hardware():
    # ── 연결 우선순위 ────────────────────────────────────────────────────────
    # 1순위: IBM Cloud (IAM API Key + CRN)
    iam_key = os.environ.get("IBM_IAM_APIKEY", "")
    crn = os.environ.get("IBM_CRN", "")
    use_quantum = os.environ.get("USE_QUANTUM_CHANNEL", "")
    quantum_token = os.environ.get("IBM_QUANTUM_TOKEN", "")

    try:
        if iam_key and crn and not use_quantum:
            service = connect_ibm_cloud(iam_key, crn)
        elif quantum_token or use_quantum:
            if not quantum_token:
                print("ERROR: IBM_QUANTUM_TOKEN not set.")
                sys.exit(1)
            service = connect_ibm_quantum(quantum_token)
        else:
            # 저장된 자격증명 시도
            print("No credentials in env vars. Trying saved account...")
            try:
                service = connect_from_saved()
            except Exception as e:
                print(f"\nERROR: No valid credentials found.\n{e}")
                print("\n설정 방법:")
                print("  [IBM Cloud IAM Key 발급]")
                print("  1. https://cloud.ibm.com -> Manage -> Access(IAM) -> API keys")
                print("  2. 'Create an IBM Cloud API key' 클릭 후 키 복사")
                print("  3. export IBM_IAM_APIKEY='<발급받은 키>'")
                print("     export IBM_CRN='crn:v1:bluemix:...'")
                sys.exit(1)
    except Exception as e:
        print(f"Connection failed: {e}")
        raise

    backend = service.backend(BACKEND_NAME)
    print(f"\nUsing backend: {backend.name} ({backend.num_qubits} qubits)")

    # ── Build circuit ─────────────────────────────────────────────────────────
    qc = QuantumCircuit(N)
    qc.x(0)
    qc.x(2)
    qc.compose(build_second_order_trotter(N, J, h, T, ZNE_R), inplace=True)
    qc_t = transpile(qc, backend=backend, optimization_level=1)
    print(f"Transpiled circuit depth: {qc_t.depth()}, Gates: {qc_t.count_ops()}")

    def real_executor(circuit) -> float:
        c = circuit.copy()
        c.measure_all()
        tc = transpile(c, backend=backend, optimization_level=1)
        sampler = Sampler(backend)
        job = sampler.run([tc], shots=SHOTS)
        print(f"  Job ID: {job.job_id()} -- waiting...")
        result = job.result()
        counts = result[0].data.meas.get_counts()
        exp_z = sum((1.0 if b[-1] == '0' else -1.0) * c for b, c in counts.items())
        return exp_z / sum(counts.values())

    print("\nRunning unmitigated job on real hardware...")
    unmitigated_val = real_executor(qc)
    print(f"Unmitigated <Z_0>: {unmitigated_val:.4f}")

    print("\nRunning ZNE-mitigated jobs on real hardware...")
    mitigated_val = apply_zne(qc, real_executor, scale_factors=[1.0, 2.0, 3.0])
    print(f"Mitigated   <Z_0>: {mitigated_val:.4f}")

    os.makedirs('results', exist_ok=True)
    output = {
        "backend": backend.name,
        "N": N, "J": J, "h": h, "T": T,
        "trotter_order": 2, "trotter_steps": ZNE_R, "shots": SHOTS,
        "unmitigated_z0": float(unmitigated_val),
        "mitigated_z0_richardson": float(mitigated_val),
    }
    with open('results/real_hardware_results.json', 'w') as f:
        json.dump(output, f, indent=4)
    print("\nSaved: results/real_hardware_results.json")


if __name__ == "__main__":
    run_on_real_hardware()
