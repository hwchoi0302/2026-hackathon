from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity

def calculate_fidelity(ideal_state, noisy_state) -> float:
    """
    Calculates the fidelity between an ideal state and a noisy state.
    
    Args:
        ideal_state: Statevector or DensityMatrix of the ideal state.
        noisy_state: Statevector or DensityMatrix of the noisy state.
        
    Returns:
        Fidelity (float between 0.0 and 1.0).
    """
    return state_fidelity(ideal_state, noisy_state)

def analyze_circuit(circuit: QuantumCircuit) -> dict:
    """
    Analyzes a Qiskit quantum circuit and extracts performance metrics.
    
    Args:
        circuit: Qiskit QuantumCircuit.
        
    Returns:
        Dictionary containing depth, total gates, multi-qubit gates, etc.
    """
    ops = circuit.count_ops()
    total_gates = sum(ops.values())
    
    depth = circuit.depth()
    
    # Calculate multi-qubit gates
    from qiskit.converters import circuit_to_dag
    dag = circuit_to_dag(circuit)
    multi_qubit_gates = sum(1 for node in dag.op_nodes() if len(node.qargs) > 1)
    
    metrics = {
        "depth": depth,
        "total_gates": total_gates,
        "multi_qubit_gates": multi_qubit_gates,
        "operations": dict(ops)
    }
    return metrics
