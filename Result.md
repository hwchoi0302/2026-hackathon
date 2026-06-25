# 1D XYZ 스핀 링 해밀토니안 Trotter 시뮬레이션 실험 결과 보고서 (non-qkd1 vs non-qkd2)

본 보고서는 1차원 주기적 이방성 XYZ 하이젠베르크 스핀 모델(1D Periodic Anisotropic XYZ Heisenberg Model)을 4큐비트 상에서 디지털 양자 시뮬레이션으로 구현하고, 두 가지 실험 설계 방식(Fixed Step-Size `non-qkd1` vs Fixed Epsilon-Precision `non-qkd2`)에 따른 시뮬레이션 정확도 및 노이즈 극복 성능을 비교 분석한 결과입니다.

---

## 1. Exact 값 (이론 참값) 계산 방법 (공통)

시뮬레이션의 정확도를 평가하기 위해서는 오차가 없는 이론상의 참값인 **Exact 값**이 필요합니다. 이 값은 클래식 컴퓨터의 선형대수 라이브러리(SciPy 등)를 사용해 **정확한 대각화(Exact Diagonalization)** 방식으로 구합니다.

1. **해밀토니안 행렬 구성**: 4개의 큐비트 시스템의 총 상태 공간 크기는 $2^4 = 16$입니다. 따라서 해밀토니안 물리량을 표현하는 전체 행렬 크기는 $16 \times 16$의 복소 행렬이 됩니다.
2. **시간 진화 연산자 계산**: 슈뢰딩거 방정식에 따라 시간 $t$에서의 상태는 다음과 같이 시간 진화 연산자 행렬을 초기 상태에 곱해 계산합니다:
   $$\lvert \psi(t) \rangle = e^{-i \hat{H} t} \lvert \psi(0) \rangle$$
   여기서 $e^{-i \hat{H} t}$ 행렬 지수함수(Matrix Exponential)를 정확하게 계산합니다.
3. **관측값 계산**: 초기 상태는 앙티페로마그네틱 Neel State $\lvert \psi(0) \rangle = \lvert 0101 \rangle$로 준비하며, 추적하려는 물리량은 0번 사이트의 단일 스핀 자화율 $\langle Z_0(t) \rangle = \langle \psi(t) \rvert Z_0 \lvert \psi(t) \rangle$입니다. 이 이론상의 기댓값을 매 시간 step마다 정밀하게 계산해 비교 기준으로 삼습니다.

---

## 2. 실험 과정 및 결과 분석 방법

1. **시간 범위 설정**: 총 진화 시간 $t$를 $0.0$부터 $2.0$까지 $0.1$ 간격으로 나눈 21개의 포인트에 대해 실험을 수행합니다.
2. **회로 생성 및 실행**:
   - 각 시간 포인트 $t$에 맞춰 Lie-Trotter(1차), Suzuki-2nd(2차), Suzuki-4th(4차) 공식을 적용하여 양자 회로를 동적으로 생성합니다.
   - 이 회로를 양자 백엔드(noiseless 시뮬레이터, FakeBrisbane 노이즈 모델 시뮬레이터, 실제 IBM Yonsei 하드웨어)에 트랜스파일하여 실행시킵니다.
3. **데이터 수집 및 메트릭 산출**:
   - 백엔드에서 관측값 $\langle Z_0(t) \rangle$ 결과를 수집합니다.
   - 참값(Exact)과의 유사성을 평가하기 위해 **RMS Error(루트 평균 제곱 오차)**, **Max Absolute Error(최대 절대 오차)** 및 **Correlation(상관 계수/충실도)** 세 가지 정량 메트릭을 도출합니다.
4. **FFT 스펙트럼 분석**:
   - 구한 진화 궤적 신호에 Hann Window를 적용하여 주파수 누설을 막고, 주파수 도메인 상의 분해능을 극대화하기 위해 4배의 Zero-Padding을 덧붙여 고해상도 FFT를 수행합니다.
   - 검출된 피크 주파수들과 해밀토니안 에너지 준위 격차(Energy Gap) 주파수들을 매핑하여 물리적 고유 진동수가 양자 회로에서 온전히 살아남았는지 검증합니다.

---

## 3. 실험 환경 및 세부 매개변수 비교

두 실험은 Trotter step 수($r$)를 결정하는 방식과 노이즈 극복 전략(Error Mitigation)에서 큰 차이가 있습니다.

| 항목 | non-qkd1 (Fixed Step Size) | non-qkd2 (Fixed Epsilon Precision) |
| :--- | :--- | :--- |
| **핵심 설계** | 시간 간격 $dt = 0.2$ 고정 | 허용 상태 오차(Infidelity) $\epsilon \le 0.05$ 고정 |
| **시간당 step 수 $r$** | 시간 $t$에 비례하여 균일 증가 ($t=2.0$ 일 때 모두 $r=10$으로 동일) | 차수별 수학적 오차 한계에 따라 동적 계산 ($t=2.0$ 일 때 Lie: 21, Suzuki-2nd: 5, Suzuki-4th: 2) |
| **최대 2-qubit 게이트 수**<br>(at $t=2.0$ ECR gate) | - Lie-Trotter: **120개**<br>- Suzuki-2nd: **240개**<br>- Suzuki-4th: **1200개** | - Lie-Trotter: **252개**<br>- Suzuki-2nd: **120개**<br>- Suzuki-4th: **240개** |
| **총 측정 횟수 (Shots)** | target standard error = 0.01 (회로당 약 10,000 shots) | target standard error = 0.01 (회로당 약 10,000 shots) |
| **Error Mitigation** | - **DD (XY4)**: 모든 노이즈/QPU 런에 적용<br>- **TREX**: QPU 런에 기본 적용<br>- ZNE 미적용 (`resilience_level=1`) | - **DD (XY4)**: 모든 노이즈/QPU 런에 적용<br>- **TREX**: QPU 런에 기본 적용<br>- **Dynamic ZNE**: 최대 2Q 게이트 수에 맞춰 자동 조절 (`resilience_level=1` or `2`) |
| **QPU 실행 모델** | Lie-Trotter, Suzuki-2nd만 수행 (4th 제외) | Lie-Trotter, Suzuki-2nd만 수행 (4th 제외) |

---

## 4. 각 실험 결과 및 오차율 비교

실험에서 얻어진 정량적 오차 메트릭의 분석 결과입니다.

### [실험 1: non-qkd1] (고정 Step Size)
이론적으로는 동일 step($r=10$)에서 고차식일수록 성능이 좋아야 하나, 노이즈 환경에서는 정반대의 현상이 일어납니다.

* **Noiseless 시뮬레이터 (이상적)**:
  - Suzuki-4th: **RMSE = 0.0080**, Fidelity = 0.9998
  - Suzuki-2nd: **RMSE = 0.0384**, Fidelity = 0.9969
  - Lie-Trotter: **RMSE = 0.0842**, Fidelity = 0.9835
  - *이상적인 환경에서는 고차 공식일수록 정확도가 월등하게 우수합니다.*
* **Real QPU 하드웨어 (`ibm_yonsei`)**:
  - Lie-Trotter: **RMSE = 0.1734**, Fidelity = 0.9311
  - Suzuki-2nd: **RMSE = 0.2685**, Fidelity = 0.8331
  - *하드웨어 노이즈 하에서는 역전이 일어납니다. Suzuki-2nd 회로(240 ECR 게이트)가 Lie(120 ECR 게이트)보다 2배 더 깊어 게이트 노이즈로 인해 결과가 심하게 뭉개졌기 때문입니다.*

### [실험 2: non-qkd2] (고정 Epsilon $\epsilon=0.05$)
오차 한계를 5% 이내로 보장하기 위해 step 수를 조정한 결과, 양자 하드웨어 상의 결과가 정반대로 뒤집힙니다.

* **Noiseless 시뮬레이터 (이상적)**:
  - Lie-Trotter ($r=21$): **RMSE = 0.0217**, Fidelity = 0.9990
  - Suzuki-4th ($r=2$): **RMSE = 0.0235**, Fidelity = 0.9988
  - Suzuki-2nd ($r=5$): **RMSE = 0.1346**, Fidelity = 0.9684
  - *Noiseless에서는 Suzuki-2nd의 RMSE가 가장 크게 나왔습니다. 하지만 이는 비정상이 아니라 최적화 결과입니다. Suzuki-2nd는 오차 한계(5%)를 아슬아슬하게 만족하는 최소한의 자원인 $r=5$ step만을 썼고, Lie는 $r=21$ step이라는 과도한 연산량을 할당받았기 때문에 생기는 자연스러운 결과입니다.*
* **Real QPU 하드웨어 (`ibm_yonsei`)**:
  - **Suzuki-2nd ($r=5$)**: **RMSE = 0.1114**, **Fidelity = 0.9780**
  - **Lie-Trotter ($r=21$)**: **RMSE = 0.2237**, **Fidelity = 0.8888**
  - ***★ 대역전 발생 ★**: 실제 QPU 환경에서 Suzuki-2nd가 Lie-Trotter를 압도적인 차이로 이겼습니다. Suzuki-2nd의 RMSE는 Lie의 반토막 수준이며, 실제 이론 곡선과 거의 일치하는 매우 깨끗한 요동(Correlation 97.8%)을 구현해 냈습니다.*

---

## 5. 두 실험의 비교 분석 및 종합 요약

### 1) 왜 non-qkd2에서 Suzuki-2nd가 실제 장비에서 압도적으로 우수한가?
* **회로 깊이의 극적인 감소**: 고정 정밀도 하에서는 차수가 높을수록 필요한 Trotter step 수가 엄청나게 줄어듭니다. $t=2.0$ 기준 Lie는 21 step이 필요해 2-qubit 게이트가 252개까지 증가했지만, Suzuki-2nd는 고작 5 step만으로 정밀도를 만족하므로 2-qubit 게이트 수가 120개에 불과했습니다.
* **에러 완화 기법(Dynamic ZNE)의 효과 극대화**: QPU의 코히어런스 한계 내에서 에러 완화가 가능하도록, 2-qubit 게이트 수가 150개 미만인 Suzuki-2nd sweep에만 **Zero Noise Extrapolation (ZNE, `resilience_level=2`)** 옵션이 자동으로 활성화되었습니다. 얕아진 회로 깊이와 강력한 ZNE의 시너지 효과 덕분에 물리적 노이즈가 성공적으로 극복되었습니다. 반면 Lie-Trotter는 회로가 너무 길어 ZNE 적용 시 게이트 노이즈 오버헤드가 더 커지므로 ZNE가 강제로 비활성화(`resilience_level=1`)되었습니다.

### 2) 핵심 결론
* **단순히 Trotter step의 크기를 고정하는 방식(non-qkd1)**은 고차 공식의 게이트 오버헤드만 부각시켜, 실제 하드웨어에서는 저차 Lie-Trotter 공식이 더 우수하다는 왜곡된 결론을 낳을 수 있습니다.
* **목표 정밀도(Fixed-Epsilon)를 기준으로 설계하는 방식(non-qkd2)**은 고차 공식이 정밀도 수렴성이 빠르다는 이점을 살려 자원(Step 수)을 획기적으로 다이어트할 수 있게 해 줍니다. 결과적으로 **회로가 더 얕아져 실제 장비에서 양자 에러 완화(DD + TREX + ZNE)의 혜택을 온전히 받으며 가장 높은 물리적 시뮬레이션 품질을 달성**할 수 있게 됩니다.
