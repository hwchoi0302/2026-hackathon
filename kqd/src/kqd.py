import numpy as np
import scipy as sp
from typing import Union, List, Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

def solve_regularized_gen_eig(
    h: np.ndarray,
    s: np.ndarray,
    threshold: float,
    k: int = 1,
    return_dimn: bool = False,
) -> Union[float, List[float], Tuple]:
    """
    Solves the generalized eigenvalue problem H c = E S c with regularization.
    Filters out the eigenvalues of S that are below the given threshold.
    """
    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = np.array(
        [vec for val, vec in zip(s_vals, s_vecs) if val > threshold]
    )
    
    # Fallback to avoid empty subspace if threshold is too high
    if len(good_vecs) == 0:
        idx = np.argmax(s_vals)
        good_vecs = np.array([s_vecs[idx]])
        
    h_reg = good_vecs.conj() @ h @ good_vecs.T
    s_reg = good_vecs.conj() @ s @ good_vecs.T
    
    eigvals, _ = sp.linalg.eigh(h_reg, s_reg)
    
    if k == 1:
        val = float(eigvals[0])
    else:
        val = [float(ev) for ev in eigvals[:k]]
        
    if return_dimn:
        return val, len(good_vecs)
    return val

def is_vacuum_eigenstate(H: SparsePauliOp) -> Tuple[bool, complex]:
    """
    Checks if the vacuum state |0>^N is an eigenstate of the Hamiltonian H.
    Returns (is_eigenstate, eigenvalue).
    """
    n_qubits = H.num_qubits
    vacuum = Statevector.from_label('0' * n_qubits)
    H_vac = vacuum.evolve(H)
    
    # Check if evolved state is proportional to the vacuum state
    val = H_vac.data[0]
    # Check if other components are close to 0
    other_norm = np.linalg.norm(H_vac.data[1:])
    is_eigen = other_norm < 1e-8
    return is_eigen, val

def build_kqd_template_circuit(
    n_qubits: int,
    state_prep_circuit: QuantumCircuit,
    H_op: SparsePauliOp,
    t_param: Parameter,
    num_trotter_steps: int = 1,
    synthesis_name: str = "lie"
) -> QuantumCircuit:
    """
    Builds the template circuit for the Efficient Hadamard Test.
    The ancilla is qubit 0, and the system qubits are 1 to n_qubits.
    """
    # 1. Controlled State Preparation
    # We construct a controlled version of the state_prep_circuit
    # Since state_prep_circuit has n_qubits, controlled_state_prep will have n_qubits + 1
    # with control at qubit 0, and system at 1..n_qubits.
    controlled_state_prep = state_prep_circuit.to_gate().control(1)
    
    # 2. Time evolution operator
    if synthesis_name == "lie":
        synthesis = LieTrotter(reps=num_trotter_steps)
    elif synthesis_name == "suzuki_2":
        synthesis = SuzukiTrotter(order=2, reps=num_trotter_steps)
    elif synthesis_name == "suzuki_4":
        synthesis = SuzukiTrotter(order=4, reps=num_trotter_steps)
    else:
        raise ValueError(f"Unknown synthesis: {synthesis_name}")
        
    evol_gate = PauliEvolutionGate(H_op, time=t_param, synthesis=synthesis)
    
    # 3. Assemble the template circuit
    qr = QuantumRegister(n_qubits + 1)
    qc = QuantumCircuit(qr)
    
    # Apply H on ancilla
    qc.h(0)
    
    # Controlled state prep (controlled by ancilla=1)
    qc.compose(controlled_state_prep, list(range(n_qubits + 1)), inplace=True)
    qc.barrier()
    
    # Uncontrolled time evolution on the system qubits (1..n_qubits)
    qc.compose(evol_gate, list(range(1, n_qubits + 1)), inplace=True)
    qc.barrier()
    
    # Controlled state prep inverse (controlled by ancilla=0)
    qc.x(0)
    qc.compose(controlled_state_prep.inverse(), list(range(n_qubits + 1)), inplace=True)
    qc.x(0)
    
    return qc
