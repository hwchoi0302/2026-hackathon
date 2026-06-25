### 개념들

- Kyylov Quantum Diagonalization >>> 추후 공부하고 적용
- ZNE >>> 적용x
- Noise learning >>> 적용 x
- Symplectic
- Gate twirling >>> DD
- PEA(Phase Esitmation Algorithm) >>> nosie learning
- OBP(Operator Back-Propagation)
- Dynamic Decoupling
- AQC

### Think

- Trotter module 분해 최적화 안되어있음
- Trotter model error 계산
  - 양자 모델 실행 전 depth, 2 qubit gate 수 계산
- 오류 정정: DD, TREX, PEA(Probablistic Error Amplification)
- Trotter 영향
  - Trotter step (Lie, Suzuki (2k))
  - Number of hamiltonian pieces
  - Circuit depth
  - error mitigation
- 오차 계산 방법
  - 이론값 비교
  - 수학적 계산

### Improvement Techniques

#### KQD (Krylov Quantum Diagonalization)

#### qDRIFT (무작위 트롤터화, Randomized Trotterization)

결정론적으로 해밀토니안의 모든 항을 순서대로 실행하는 대신, 각 상호작용 항의 세기 (계수) 에 비례하는 확률을 부여하여 회로에 들어갈 게이트를 무작위로 샘플링합니다. 큐비트 수나 항의 개수가 많아져도 양자 회로 깊이를 일정 수준 이하로 고정할 수 있어 노이즈가 많은 현재 하드웨어에 매우 유리합니다.

#### VQTE (변분 양자 시간 진화, Variational Quantum Time Evolution)

깊은 시간 진화 회로 $e^{-iHt}$ 를 직접 양자 하드웨어에 올리는 대신, 매우 얕은 깊이의 튜닝 가능한 매개변수화 양자 회로 (PQC) 에 시간 진화 궤적을 투영시킵니다. 양자 컴퓨터는 얕은 회로만 실행하여 측정하고, 고전 컴퓨터가 변분 원리를 이용해 다음 시간 스텝의 매개변수를 업데이트하므로 회로 깊이의 누적을 원천적으로 차단합니다.

#### Symmetry Protection (대칭성 보호)

대상 시스템의 물리적 대칭성 (예: 전체 스핀량 보존, 패리티 보존) 을 활용하여, 회로 중간에 의도적으로 부호를 뒤집는 연산자를 대칭적으로 삽입합니다. 양자 회로 깊이를 늘리지 않고도 고차원의 Trotter 오차 항들을 수학적으로 서로 상쇄시키는 고급 컴파일 기법입니다.

#### QSP (양자 신호 처리, Quantum Signal Processing) 및 Qubitization

향후 오류 내성 양자 컴퓨터 (FTQC) 시대의 표준이 될 최적의 알고리즘입니다. Trotter 방식처럼 시간을 잘게 쪼개어 근사하는 것이 아니라, 보조 큐비트를 사용해 해밀토니안 전체를 하나의 거대한 유니터리 블록으로 인코딩한 뒤 다항식 최적화를 통해 오차를 지수적으로 억제합니다.

### Trotter step, \Delta t 설정

Trotter 스텝 수(0.2 고정)에 대한 리뷰 및 수학적 최적화 방법코드 상태 리뷰: 실험 리포트와 구현 방식을 볼 때, 단일 시간 간격 $\Delta t = 0.2$를 고정 상수처럼 취급하고 전체 시간 $t$에 따라 스텝 수 $r$을 선형적으로 늘린 것은 전형적인 휴리스틱(경험적) 접근이 맞습니다.

이는 당시 사용했던 하드웨어(ibm_yonsei 등)가 견딜 수 있는 게이트 깊이의 한계를 실험적으로 타협한 결과일 것입니다.논문 기반의 최적 스텝 수 결정법:"A Theory of Trotter Error" 논문의 교환자(Commutator) 스케일링 이론에 따르면, 진정한 최적화는 단순히 상수 $\Delta t$ 를 고정하는 것이 아니라, '알고리즘 오차'와 '하드웨어 물리적 오차'의 상충(Trade-off) 관계를 하나의 비용 함수로 묶어 최소화하는 과정을 거쳐야 합니다.

수학적 Trotter 오차 ($E_{math}$): 논문에 따라 1차 Lie-Trotter의 오차 상한은 대략 $c_1 \frac{t^2}{r}$ ( $c_1$은 교환자의 스펙트럼 놈)에 비례하여 감소합니다.하드웨어 누적 에러 ($E_{noise}$): CNOT 게이트 1개당 평균 에러율을 $\epsilon_{cnot}$ 이라 할 때, 1스텝에 들어가는 CNOT 개수를 $N$ 이라 하면, 전체 물리적 에러는 대략 $c_2 \cdot r \cdot N \cdot \epsilon_{cnot}$ 에 비례하여 선형(또는 지수)적으로 증가합니다.

최적화 수식: 총 오차 $E_{total}(r) = c_1 \frac{t^2}{r} + c_2 \cdot r \cdot N \cdot \epsilon_{cnot}$ 라는 비용 함수를 구성합니다.결정: 시뮬레이션을 돌리기 전, Qiskit의 backend.properties()를 호출해 해당 날짜의 실제 하드웨어 에러율($\epsilon_{cnot}$) 데이터를 받아옵니다. 이 값을 위 식에 대입한 후, $r$에 대해 미분하여 최솟값을 갖는 지점($r^*$)을 찾습니다. 그 결과 산출된 $r^*$로 회로를 구성하는 것이 이론과 하드웨어 스펙을 모두 고려한 가장 완벽한 동적 최적화입니다.

## 개발 로드맵

### Symmetry Protection 삽입 (난이도 하, 효과 최상):
 코드에 Z-패리티 게이트를 샌드위치처럼 넣습니다. IBM 하드웨어에서는 가상 게이트라 노이즈 추가 없이 에러율만 낮춥니다.
 ### 최적 스텝 수(r) 탐색기 구현 (난이도 중): 
 하드웨어 노이즈 파라미터와 이론적 Trotter 오차 수식을 결합해, 임의로 스텝을 0.2로 고정하지 않고 최적의 $r$ 값을 자동으로 찾아주는 함수를 만듭니다.
 ### KQD + Lie-Trotter 모듈 작성 (난이도 상): 
 $t=2.0$까지 길게 늘어지는 회로를 버리고, $\Delta t$ 단위의 짧은 Trotter 회로 기댓값들을 측정한 뒤 고전 행렬 대각화(Eigenvalue solver)로 에너지를 추출하는 코드를 구현합니다.