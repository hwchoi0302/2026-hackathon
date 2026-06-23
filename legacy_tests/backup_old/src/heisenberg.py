import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import expm


def get_heisenberg_hamiltonian(N: int, J: float, h: float) -> np.ndarray:
    """
    Returns the matrix representation of the 1D periodic Heisenberg Hamiltonian.

    H = J * sum_{i=0}^{N-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
      + h * sum_{i=0}^{N-1} Z_i

    with periodic boundary conditions (i+1 mod N).

    Note on Qiskit Pauli convention:
      Pauli string ordering is right-to-left for qubit indices.
      e.g., 'ZI' acts as Z on qubit 0, I on qubit 1.
      So for qubit index i, we build a list of 'I' of length N,
      set index i to the desired Pauli, then reverse before joining.
    """
    from qiskit.quantum_info import Pauli

    H_matrix = np.zeros((2**N, 2**N), dtype=complex)

    # Two-body interaction terms: XX + YY + ZZ
    for i in range(N):
        j = (i + 1) % N

        # XX term
        xx_str = ['I'] * N
        xx_str[i] = 'X'
        xx_str[j] = 'X'
        H_matrix += J * Pauli(''.join(xx_str[::-1])).to_matrix()

        # YY term
        yy_str = ['I'] * N
        yy_str[i] = 'Y'
        yy_str[j] = 'Y'
        H_matrix += J * Pauli(''.join(yy_str[::-1])).to_matrix()

        # ZZ term
        zz_str = ['I'] * N
        zz_str[i] = 'Z'
        zz_str[j] = 'Z'
        H_matrix += J * Pauli(''.join(zz_str[::-1])).to_matrix()

    # Single-body external field terms: h * sum Z_i
    # NOTE: This must be outside the bond loop to avoid double-counting.
    for i in range(N):
        z_str = ['I'] * N
        z_str[i] = 'Z'
        H_matrix += h * Pauli(''.join(z_str[::-1])).to_matrix()

    return H_matrix


def exact_evolution(H: np.ndarray, t: float) -> np.ndarray:
    """
    Calculates the exact unitary time evolution matrix U(t) = exp(-i H t).
    Uses scipy.linalg.expm for matrix exponentiation.
    """
    return expm(-1j * H * t)


def build_first_order_trotter(N: int, J: float, h: float, time: float, steps: int) -> QuantumCircuit:
    """
    Builds a First-Order (Lie-Trotter) Trotterized time evolution circuit.

    Approximation: U(dt) ~ [prod_i exp(-i J dt XX_i) exp(-i J dt YY_i) exp(-i J dt ZZ_i)]
                            * [prod_i exp(-i h dt Z_i)]

    where dt = time / steps. Global error is O(dt) = O(T/r).

    Gate angles:
      RXX(theta) = exp(-i theta/2 * XX)  -->  theta = 2*J*dt  for exp(-i J*dt XX)
      RYY(theta) = exp(-i theta/2 * YY)  -->  theta = 2*J*dt
      RZZ(theta) = exp(-i theta/2 * ZZ)  -->  theta = 2*J*dt
      RZ(phi)    = exp(-i phi/2 * Z)     -->  phi   = 2*h*dt  for exp(-i h*dt Z)
    """
    qc = QuantumCircuit(N)
    dt = time / steps

    for _ in range(steps):
        # 2-qubit bond interactions
        for i in range(N):
            j = (i + 1) % N
            qc.rxx(2 * J * dt, i, j)
            qc.ryy(2 * J * dt, i, j)
            qc.rzz(2 * J * dt, i, j)

        # Single-qubit external field
        for i in range(N):
            qc.rz(2 * h * dt, i)

    return qc


def build_second_order_trotter(N: int, J: float, h: float, time: float, steps: int) -> QuantumCircuit:
    """
    Builds a Second-Order (Suzuki-Trotter) Trotterized time evolution circuit.

    The Hamiltonian is decomposed into individual terms:
      bond terms: H_{bond,i} = J*(XX + YY + ZZ)_{i,i+1}  for i in 0..N-1
      field terms: H_{field,i} = h*Z_i                    for i in 0..N-1

    The 2nd-order Suzuki formula arranges these as a palindromic (symmetric) product:
      U(dt) = exp(-i h_0 dt/2) * exp(-i h_1 dt/2) * ... * exp(-i h_{K-1} dt/2)
              * exp(-i h_{K-1} dt/2) * ... * exp(-i h_0 dt/2)

    i.e., each term is applied at half the step size in forward order,
    then again at half the step size in REVERSED order. This symmetry
    cancels the O(dt^2) local error term, reducing it to O(dt^3) per step
    and O(dt^2) global error -- one order better than first-order Trotter.

    Note: In the forward+backward palindrome, adjacent dt/2 blocks from
    the same term between consecutive Trotter steps do NOT automatically
    fuse unless explicitly optimized (circuit folding). Here we keep the
    full palindromic structure within each step for clarity.
    """
    qc = QuantumCircuit(N)
    dt = time / steps

    for _ in range(steps):
        # Forward sweep: all bond terms at dt/2, then all field terms at dt/2
        for i in range(N):
            j = (i + 1) % N
            qc.rxx(2 * J * (dt / 2), i, j)
            qc.ryy(2 * J * (dt / 2), i, j)
            qc.rzz(2 * J * (dt / 2), i, j)
        for i in range(N):
            qc.rz(2 * h * (dt / 2), i)

        # Backward sweep (reversed order): all field terms at dt/2, then all bond terms at dt/2
        for i in reversed(range(N)):
            qc.rz(2 * h * (dt / 2), i)
        for i in reversed(range(N)):
            j = (i + 1) % N
            qc.rxx(2 * J * (dt / 2), i, j)
            qc.ryy(2 * J * (dt / 2), i, j)
            qc.rzz(2 * J * (dt / 2), i, j)

    return qc
