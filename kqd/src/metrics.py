from qiskit import QuantumCircuit

def analyze_transpiled_circuit(circuit: QuantumCircuit, backend, optimization_level: int = 2) -> dict:
    """
    Transpiles the circuit and analyzes the transpiled result metrics.
    
    Args:
        circuit: The base QuantumCircuit.
        backend: The target Qiskit backend.
        optimization_level: Transpiler optimization level (0, 1, 2, or 3).
        
    Returns:
        Dictionary with transpiled circuit metrics (depth, total_gates, two_qubit_gates, operations).
    """
    from qiskit import transpile
    
    # Transpile the circuit
    transpiled_qc = transpile(circuit, backend=backend, optimization_level=optimization_level)
    ops = transpiled_qc.count_ops()
    
    metrics = {
        "depth": transpiled_qc.depth(),
        "total_gates": sum(ops.values()),
        "two_qubit_gates": transpiled_qc.num_nonlocal_gates(),
        "operations": dict(ops)
    }
    return metrics
