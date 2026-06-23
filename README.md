# 1D Anisotropic XYZ Spin Ring Hamiltonian Simulation & Spectral Analysis

이 저장소는 1차원 주기적 이방성 XYZ 스핀 고리(1D Periodic Anisotropic XYZ Spin Ring) 모델의 시간 진화(Time-evolution)를 양자 컴퓨터 상에서 시뮬레이션하고, 수집된 데이터를 스펙트럼 분석(FFT)하여 에너지 갭(Energy Gaps)을 도출하는 오픈소스 프레임워크입니다.

Qiskit 2.x Primitives(EstimatorV2) 및 최신 에러 완화(Error Mitigation) 기술을 탑재하여 Noiseless, Noisy Fake Backend, 그리고 IBM Quantum 실제 하드웨어(`ibm_yonsei`) 실행까지 완벽하게 지원합니다.

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

## 2. 프로젝트 폴더 구조

프로젝트의 핵심 라이브러리와 실행 파일, 테스트 스크립트들이 다음과 같이 체계적으로 정리되어 있습니다.

```
2026-hackathon/
│
├── src/                      # 핵심 기능 모듈 (Core Library)
│   ├── hamiltonian.py        # 1D Ring Topology XYZ Hamiltonian 생성
│   ├── circuit_builder.py    # Lie-Trotter (1st), Suzuki-Trotter (2nd, 4th) 회로 구성
│   ├── mitigation.py         # 로컬 DD 패스 매니저 및 IBM Runtime DD/TREX 옵션 바인딩
│   ├── execution.py          # Noiseless, Fake, Real 백엔드용 EstimatorV2 빌더
│   ├── analysis.py           # Pure Python/SciPy 기반 FFT 스펙트럼 분석 및 피크 검출
│   └── metrics.py            # 트랜스파일 후 회로 사양(Depth, ECR 수) 분석
│
├── scripts/                  # 메인 구동 제어판 (Runnable Execution)
│   └── run_harness.py        # 시뮬레이션 파이프라인 구동, 플롯 출력, JSON 세이브
│
├── legacy_tests/             # 검증용 테스트 스크립트 및 유틸리티 모음 (임시 파일 관리)
│   ├── compute_exact.py      # 정확 대각화(Exact Diagonalization) 기반 이론값 계산
│   ├── generate_report.py    # 결과 비교 그래프 플로팅 및 simulation_report.tex 생성
│   └── fetch_real_results.py # 노트북 전원을 꺼도 되는 오프라인용 실장비 결과 수집기
│
└── results/                  # 데이터 결과물 (JSON, PNG, TEX, PDF)
```

---

## 3. 핵심 구현 및 하드웨어 최적화 기법

### 일괄 배치 실행 (Batch Execution)
실제 양자 컴퓨터(`ibm_yonsei`)에 회로를 제출할 때, 각 시간 포인트별로 20번의 API 호출을 수행하면 극심한 큐(Queue) 대기 지연이 발생합니다. 본 프로젝트는 **모든 시간 포인트의 회로와 관측값을 하나의 단일 잡(Job)으로 묶어 배치 제출**하도록 최적화되었습니다. 이로 인해 대기 큐 대기 시간이 **40배 단축**되는 효과를 얻었습니다.

### 에러 완화 (Error Mitigation)
1. **Dynamic Decoupling (DD):** 2큐비트 게이트 연산 중에 아무 작동도 하지 않고 대기하는 큐비트들에 탈위상화(Dephasing) 방지를 위한 $X-Y-X-Y$ (XY4 sequence) 펄스를 주입하여 결맞음 시간을 강제로 연장시킵니다.
2. **TREX (Twirled Readout Error eXtinction):** 양자 상태 측정 시 겪는 리드아웃 노이즈를 측정 전 임의 게이트(twirling) 및 사후 처리를 통해 보정합니다.
3. **ZNE (Zero Noise Extrapolation) 제외 사유:** ZNE를 사용하려면 노이즈 레벨 확장을 위해 회로를 3~5배 강제로 복제해야 합니다. 이는 2큐비트 게이트 개수를 폭증시켜 NISQ 장비의 결맞음 한계를 아득히 넘어가 신호를 완전히 뭉개버리므로 **ZNE를 차단하고 DD + TREX만 적용하는 것이 실장비 주파수 복원에 훨씬 유리합니다.**

---

## 4. 모의 시뮬레이션 결과 및 분석

### A. Noiseless (이상적 Trotter 오차 비교)
이론값(Exact Diagonalization)과 이상적 환경에서의 시뮬레이션 오차 비교입니다 ($t=2.0$, $r=10$ 기준).
- **1st-order Lie-Trotter:** RMS Error = 0.0843 | Correlation = 0.9834
- **2nd-order Suzuki-Trotter:** RMS Error = 0.0385 | Correlation = 0.9970
- **4th-order Suzuki-Trotter:** RMS Error = 0.0080 | Correlation = 0.9999
*수학적으로 차수가 높은 Suzuki-4th가 완벽한 이론 수렴성을 보여줍니다.*

### B. Noisy (FakeBrisbane 노이즈 및 회로 깊이의 Trade-off)
실제 하드웨어 에러 모델 하에서 오차 비교입니다.
- **1st-order Lie-Trotter:** RMS Error = **0.1634** | Correlation = **0.9501**
- **2nd-order Suzuki-Trotter:** RMS Error = 0.2678 | Correlation = 0.8411
- **4th-order Suzuki-Trotter:** RMS Error = 0.3705 | Correlation = 0.6189

> [!IMPORTANT]
> **결정적인 물리적 역전 현상:**
> 노이즈가 없는 가상 환경에서는 Suzuki-4th가 가장 뛰어났으나, 실제 노이즈 환경에서는 **Suzuki-4th 회로가 너무 깊어(ECR 게이트 600개) 신호가 완전히 감쇄**되었습니다. 
> 반면, 가장 얕은 **Lie-Trotter(ECR 게이트 600개 대비 60개로 10배 짧음)가 노이즈 하에서 기하급수적으로 더 높은 정확도(상관계수 0.95)를 보존**합니다.

---

## 5. 실행 방법

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 1. 로컬 모의 시뮬레이션 구동 (Noiseless / Fake)
```bash
# 이상적 상태 시뮬레이터 구동
python scripts/run_harness.py --backend noiseless --models lie,suzuki_2,suzuki_4

# 노이즈 모델 시뮬레이터 구동 (DD 적용)
python scripts/run_harness.py --backend fake --models lie,suzuki_2,suzuki_4
```

### 2. 실제 양자 컴퓨터 실행 (ibm_yonsei)
실물 하드웨어 장비에서는 성능 저하가 극심한 Suzuki-4th를 제외하고 **Lie와 Suzuki-2nd만 일괄 배치**로 연산하여 큐 대기 오버헤드를 아낍니다.
```bash
python scripts/run_harness.py --backend real --models lie,suzuki_2 --enable-dd --dd-sequence XY4 --enable-trex
```

---

## 6. 💻 노트북 종료 대처 가이드 (오프라인 모니터링)

IBM Quantum 실제 장비의 대기열(QUEUED) 시간이 너무 길어져 노트북 전원을 끄고 싶을 때는 **그냥 끄고 퇴근하셔도 됩니다.** 
클라우드 상에 제출된 작업은 자동으로 계속 진행되며, 나중에 노트북을 다시 켜고 아래 명령어를 단 한 번만 실행하면 됩니다.

```bash
python legacy_tests/fetch_real_results.py
```

이 스크립트는 다음 작업을 지능적으로 대행합니다:
1. IBM Cloud에서 첫 번째 작업(`Lie-Trotter` Job)의 완료 여부를 확인하고 결과를 회수합니다.
2. 만약 노트북 종료로 인해 누락되었던 **두 번째 작업(`Suzuki-2nd`)을 즉석에서 새로 구성하여 자동으로 IBM Cloud에 이어서 제출**해 줍니다.
3. 모든 연산 결과가 모이면 FFT 물리 스펙트럼 변환을 거쳐 `results/simulation_report.tex` 및 `simulation_report.pdf` 보고서와 비교 플롯 그래프를 최신 데이터로 일괄 자동 빌드하여 완벽히 재생성합니다.
