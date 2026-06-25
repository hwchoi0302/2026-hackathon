from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

def build_trotter_circuit(
    hamiltonian: SparsePauliOp,
    time: float,
    steps: int,
    model_name: str = "suzuki_2"
) -> QuantumCircuit:
    """
    Builds the Trotterized time-evolution circuit.
    
    Args:
        hamiltonian: SparsePauliOp representing the system Hamiltonian.
        time: Total time of evolution.
        steps: Number of Trotter steps.
        model_name: One of 'lie' (1st order), 'suzuki_2' (2nd order Suzuki), 'suzuki_4' (4th order Suzuki).
        
    Returns:
        QuantumCircuit containing the evolution.
    """
    if model_name == "lie":
        synthesis = LieTrotter(reps=steps)
    elif model_name == "suzuki_2":
        synthesis = SuzukiTrotter(order=2, reps=steps)
    elif model_name == "suzuki_4":
        synthesis = SuzukiTrotter(order=4, reps=steps)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Must be 'lie', 'suzuki_2', or 'suzuki_4'.")
        
    evolution_gate = PauliEvolutionGate(hamiltonian, time=time, synthesis=synthesis)
    
    num_qubits = hamiltonian.num_qubits
    qc = QuantumCircuit(num_qubits)
    qc.append(evolution_gate, range(num_qubits))
    
    return qc
