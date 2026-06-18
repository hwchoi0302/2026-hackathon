import numpy as np
import mitiq
from qiskit import QuantumCircuit
from typing import Callable, List, Optional

def apply_zne(
    circuit: QuantumCircuit,
    executor: Callable[[QuantumCircuit], float],
    scale_factors: List[float] = [1.0, 2.0, 3.0],
    factory: Optional[mitiq.zne.inference.Factory] = None,
    folding_method: str = "global"
) -> float:
    """
    Applies Zero Noise Extrapolation (ZNE) using Mitiq.
    
    Args:
        circuit: The base Qiskit QuantumCircuit to evaluate.
        executor: A function that takes a Qiskit QuantumCircuit, executes it, 
                  and returns an expectation value (float).
        scale_factors: List of noise scale factors to evaluate.
        factory: Mitiq extrapolation factory (e.g., RichardsonFactory, LinearFactory). 
                 Defaults to RichardsonFactory if None.
        folding_method: "global" (fold_global) or "local" (fold_gates_at_random).
        
    Returns:
        The zero-noise extrapolated expectation value.
    """
    if factory is None:
        factory = mitiq.zne.inference.RichardsonFactory(scale_factors)
        
    # Choose folding function
    if folding_method == "global":
        fold_func = mitiq.zne.scaling.fold_global
    elif folding_method == "local":
        fold_func = mitiq.zne.scaling.fold_gates_at_random
    else:
        raise ValueError("folding_method must be 'global' or 'local'")

    # Execute ZNE
    mitigated_value = mitiq.zne.execute_with_zne(
        circuit,
        executor,
        factory=factory,
        scale_noise=fold_func
    )
    
    return mitigated_value