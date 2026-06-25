# 1D Anisotropic XYZ Spin Ring Hamiltonian Simulation & Spectral Analysis

이 저장소는 1차원 주기적 이방성 XYZ 스핀 고리(1D Periodic Anisotropic XYZ Spin Ring) 모델의 시간 진화(Time-evolution) 및 양자 크릴로프 부공간 대각화(Quantum Krylov Subspace Diagonalization)를 양자 컴퓨터 상에서 시뮬레이션하고, 물리적 성질을 규명하는 오픈소스 프레임워크입니다.

Qiskit 2.x Primitives (EstimatorV2) 및 최신 에러 완화(Error Mitigation) 기술을 탑재하여 Noiseless, Noisy Fake Backend, 그리고 IBM Quantum 실제 하드웨어(`ibm_yonsei`) 실행까지 완벽하게 지원합니다.

---

## 1. 물리 모델 및 이론

### Hamiltonian
우리는 $N=4$개의 큐비트(스핀)로 구성된 1차원 주기적 경계 조건(Periodic Boundary Conditions)의 XYZ 모델을 시뮬레이션합니다. 해밀토니안은 다음과 같이 정의됩니다.

$$ \hat{H} = \sum_{i=0}^{N-1} \left( J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1} \right) + h \sum_{i=0}^{N-1} Z_i $$

여기서 결합 계수들은 이방성(Anisotropic)을 가집니다.
- $J_x = 1.0$, $J_y = 0.8$, $J_z = 0.5$
- 횡자장(Transverse field) 강도 $h = 0.5$
- 주기적 경계 조건 적용: $i+1 \pmod N$ (스핀 링 구조)

### 초기 상태 및 관측값
- **초기 상태 (Initial State):** 반강자성 Néel State $|\psi(0)\rangle = |0101\rangle$ (Qiskit Little-endian 기준, 0번 및 2번 큐비트가 $|1\rangle$)
- **관측값 (Observable):** 0번 큐비트의 단일 사이트 자화율 $\langle Z_0(t) \rangle$

---

## 2. 프로젝트 폴더 및 파일 구조 설명

처음 저장소를 방문하는 사용자도 각 모듈의 역할을 한눈에 이해할 수 있도록 컴포넌트별로 모듈과 핵심 함수들을 정리하였습니다.

### 📂 `src/` (핵심 시뮬레이션 모듈)
Trotter 시간 진화 시뮬레이션을 구현하는 핵심 라이브러리 모듈들입니다.
* [hamiltonian.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/src/hamiltonian.py)
  * **역할**: 시뮬레이션 대상인 XYZ Hamiltonian 연산자 생성.
  * **주요 함수**:
    * `get_xyz_hamiltonian(num_qubits, jx, jy, jz, h)`: 결합 상수와 횡자장 강도를 입력받아 주기적 경계 조건(PBC)을 적용한 XYZ Hamiltonian을 Qiskit의 `SparsePauliOp` 객체로 빌드합니다.
* [circuit_builder.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/src/circuit_builder.py)
  * **역할**: 다양한 차수의 Trotter-Suzuki 시간 진화 양자 회로 설계.
  * **주요 함수**:
    * `build_trotter_circuit(hamiltonian, time, steps, model_name)`: 1차 Lie-Trotter(`lie`), 2차 Suzuki-Trotter(`suzuki_2`), 4차 Suzuki-Trotter(`suzuki_4`) 공식을 사용해 진화 시간 $t$와 step 수 $r$에 따른 유니터리 연산 양자 회로를 조립합니다.
* [mitigation.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/src/mitigation.py)
  * **역할**: 에러 극복을 위한 로컬/실행 레벨 에러 완화 설정.
  * **주요 함수**:
    * `apply_local_dd(circuit, backend, dd_sequence_type)`: 로컬 시뮬레이션용 Dynamic Decoupling(DD) 컴파일러. 회로의 유휴 스케줄(slack) 구간에 XY4 또는 XX 시퀀스 펄스를 삽입하여 결맞음 에러를 억제합니다.
    * `configure_runtime_mitigation_options(options, enable_dd, dd_sequence, enable_trex)`: Qiskit Runtime EstimatorOptions에 IBM QPU의 내장 DD 및 TREX(Twirled Readout Error eXtinction) 측정 에러 완화 옵션을 바인딩합니다.
* [execution.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/src/execution.py)
  * **역할**: 백엔드 종류(Aer 시뮬레이터, Fake 노이즈 모델, QPU 실장비)에 따른 EstimatorV2 관리.
  * **주요 함수**:
    * `get_estimator(mode, backend_instance, enable_dd, dd_sequence, enable_trex, resilience_level)`: 백엔드 상태에 대응하는 `EstimatorV2` 객체를 반환하고 `resilience_level` 옵션을 설정합니다.
* [analysis.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/src/analysis.py)
  * **역할**: 시뮬레이션 결과 데이터 신호 처리 및 푸리에 변환 분석.
  * **주요 함수**:
    * `perform_fft_analysis(times, values, window_type, zero_padding_factor)`: 시간축 기댓값 신호에 Hann 윈도우를 씌우고 4배의 Zero-Padding을 더해 고해상도 FFT(고속 푸리에 변환) 분석을 수행하여 피크 주파수를 찾아냅니다.
* [metrics.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/src/metrics.py)
  * **역할**: 양자 회로 특성 정보 추출.
  * **주요 함수**:
    * `count_gates(circuit)`: 트랜스파일이 완료된 회로의 실제 Depth와 2-qubit 게이트(ECR/CX) 개수를 반환하여 회로 품질을 측정합니다.

### 📂 `scripts/` (구동 스크립트)
* [run_harness.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/scripts/run_harness.py)
  * **역할**: `non-qkd1` (고정 Step-Size $dt=0.2$) 실험의 Noiseless/Fake/Real 백엔드 통합 구동기.
* [run_harness_non_qkd2.py](file:///home/hyunwoo/workspace/hackathon/2026-hackathon-prep/scripts/run_harness_non_qkd2.py)
  * **역할**: `non-qkd2` (고정 Epsilon $\epsilon=0.05$) 최적화 실험의 스윕 실행, 에러 완화 적용 및 물리 분석 데이터 도출 스크립트.

### 📂 `kqd/` (Krylov Quantum Diagonalization - KQD 실험 폴더)
양자 크릴로프 부공간 대각화(QKSD) 기법을 모듈화하여 설계한 독립적인 폴더입니다.
* `kqd/src/kqd.py`
  * **역할**: KQD 알고리즘을 수행하기 위한 Hadamard Test 빌더 및 클래식 정규화 솔버 구현.
  * **주요 함수**:
    * `build_kqd_template_circuit(...)`: Efficient Hadamard Test를 통해 시간 진화 상태 간의 Overlap(중첩도)을 구하기 위한 기본 템플릿 회로를 설계합니다.
    * `solve_regularized_gen_eig(h, s, threshold, k)`: 노이즈가 낀 중첩 행렬 $S$와 해밀토니안 행렬 $H$의 차원 붕괴를 예방하기 위해 고유값 필터링 기반 임계값 정규화(Regularization)를 거쳐 일반화 고유값 문제 $H c = E S c$를 풉니다.
* `kqd/scripts/run_kqd.py`: KQD 시뮬레이션 구동 및 Overlap 행렬 생성기.
* `kqd/scripts/compare_kqd_std.py`: 기존 Trotter 시간 진화 방식과 KQD 대각화 방식의 정밀도 및 자원 소모량 비교 분석.
* `kqd/scripts/analysis_plots.py`: 측정 데이터 행렬 가시화 및 기저 상태(Ground State) 물리 분석 플로팅 스크립트.

---

## 3. 세 가지 실험 특징 및 에러 완화 비교

저장소에서 구현하고 있는 세 가지 핵심 실험 방식의 차이점을 요약합니다.

### 1️⃣ `non-qkd1` (Fixed Step-Size Trotter Simulation)
- **핵심 원리**: 시간 간격 $dt = 0.2$를 물리적으로 고정하여 시간이 증가함에 따라 Trotter Step 수 $r$을 선형적으로 늘립니다 ($r = \text{round}(t / 0.2)$). 이에 따라 시간 $t=2.0$에서 모든 모델이 동일하게 $r=10$ step을 시뮬레이션합니다.
- **2-qubit 게이트 수 (at $t=2.0$)**: Lie-Trotter 120개, Suzuki-2nd 240개, Suzuki-4th 1200개.
- **에러 완화**: Dynamic Decoupling (DD, XY4 Sequence) 및 Readout Mitigation (TREX) 적용. (ZNE는 미적용, `resilience_level=1`).
- **하드웨어 결과**: 노이즈가 없는 이론상으로는 Suzuki-4th가 정확했으나, 실제 하드웨어 QPU에서는 Suzuki-2nd 이상 고차 공식의 게이트 깊이가 너무 깊어져 노이즈 감쇄 효과로 인해 오히려 **가장 얕은 1차 Lie-Trotter 공식이 노이즈 하에서 가장 정확**하게 측정되었습니다.

### 2️⃣ `non-qkd2` (Fixed Epsilon-Precision Trotter Simulation)
- **핵심 원리**: 각 차수 공식의 수학적 오차 한계 스케일링을 기반으로, 매 시간 지점 $t$마다 목표 오차율(Infidelity) $\epsilon \le 0.05$를 충족하는 최소 step 수 $r$을 동적으로 할당합니다. 이에 따라 $t=2.0$에서 Lie: 21 steps, Suzuki-2nd: 5 steps, Suzuki-4th: 2 steps가 요구됩니다.
- **2-qubit 게이트 수 (at $t=2.0$)**: Lie-Trotter 252개, Suzuki-2nd 120개, Suzuki-4th 240개.
- **에러 완화**: DD (XY4) + TREX + **Dynamic ZNE**. 2-qubit 게이트 수 150개 미만의 경우 에러 완화 능력을 극대화하기 위해 `resilience_level=2` (ZNE + TREX)를 자동 부여합니다. 이에 따라 **Suzuki-2nd만 ZNE를 적용받고**, Lie-Trotter는 게이트 개수가 150개를 초과하여 ZNE 없이(`resilience_level=1`) 동작합니다.
- **하드웨어 결과**: Suzuki-2nd는 고차 공식의 빠른 수렴성 덕분에 필요한 step 수($r=5$)가 획기적으로 낮아져 회로가 가장 얕았고, ZNE까지 맞물리면서 **QPU 하드웨어 상에서 RMS Error 0.1114 및 Correlation 97.80%라는 가장 압도적인 성능**을 보였습니다.

### 3️⃣ `qkd` (Krylov Quantum Diagonalization)
- **핵심 원리**: 시스템을 긴 시간 동안 직접 양자 장비에서 물리적으로 진화시키는 대신, 얕은 깊이의 회로 상에서 효율적인 Hadamard Test를 수행하여 Krylov Subspace 내의 중첩 행렬 $S$ 및 해밀토니안 행렬 $H$ 성분을 측정합니다. 수집된 행렬 데이터를 고전 컴퓨터로 전달하여 일반화 고유값 관계식을 풀어서 바닥 상태 에너지 및 고유 진동수를 알아냅니다.
- **회로 깊이 및 자원 특징**:
  * **장점**: 장시간 진화 회로 대비 회로의 최고 깊이(Depth)가 얕게 유지되어 하드웨어 결맞음 시간(Coherence time)을 초과하지 않고 정밀한 스펙트럼 분석이 가능합니다.
  * **단점**: 노이즈로 인해 중첩 행렬 $S$가 비가역적으로 붕괴될 수 있으므로, 클래식 포스트 프로세서에 특이값 차단 정규화(SVD Regularization Solver) 처리가 필수적입니다.

---

## 4. 실행 방법

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 1️⃣ `non-qkd1` (Fixed Step-Size) 실행
```bash
# 로컬 noiseless 구동
python scripts/run_harness.py --backend noiseless --models lie,suzuki_2,suzuki_4

# 로컬 noisy (FakeBrisbane + DD) 구동
python scripts/run_harness.py --backend fake --models lie,suzuki_2,suzuki_4

# QPU 실장비 (ibm_yonsei + DD + TREX) 구동
python scripts/run_harness.py --backend real --models lie,suzuki_2 --enable-dd --enable-trex
```

### 2️⃣ `non-qkd2` (Fixed Epsilon-Precision) 실행
```bash
# 로컬 noiseless 구동
python scripts/run_harness_non_qkd2.py --backend noiseless --models lie,suzuki_2,suzuki_4

# 로컬 noisy (FakeBrisbane + DD) 구동
python scripts/run_harness_non_qkd2.py --backend fake --models lie,suzuki_2,suzuki_4

# QPU 실장비 (ibm_yonsei + DD + TREX + Dynamic ZNE) 구동
python scripts/run_harness_non_qkd2.py --backend real --models lie,suzuki_2 --enable-dd --enable-trex
```

### 3️⃣ `qkd` (Krylov Subspace Diagonalization) 실행
```bash
# KQD 데이터 측정 양자 회로 구동
python kqd/scripts/run_kqd.py

# Trotter 방식과의 비교 시뮬레이션 실행 및 데이터 비교 플롯 빌드
python kqd/scripts/compare_kqd_std.py
python kqd/scripts/analysis_plots.py
```
