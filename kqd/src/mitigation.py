from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
from qiskit import transpile, QuantumCircuit

def apply_local_dd(
    circuit: QuantumCircuit, 
    backend, 
    dd_sequence_type: str = "XX",
    optimization_level: int = 1
) -> QuantumCircuit:
    """
    Applies Dynamic Decoupling locally to a circuit for simulator execution.
    
    Args:
        circuit: The target QuantumCircuit.
        backend: The target backend whose target durations will be used for scheduling.
        dd_sequence_type: Type of sequence ('XX' or 'XY4').
        optimization_level: Transpiler optimization level.
        
    Returns:
        Transpiled circuit with DD pulses inserted.
    """
    # 1. Transpile to target basis gates and backend coupling map
    transpiled_qc = transpile(circuit, backend=backend, optimization_level=optimization_level)
    
    # 2. Extract timing/durations
    try:
        durations = backend.target.durations()
    except Exception:
        # Fallback if target durations are not available (e.g. basic noiseless simulator)
        return transpiled_qc
        
    # 3. Create sequence
    if dd_sequence_type == "XX":
        dd_sequence = [XGate(), XGate()]
    elif dd_sequence_type == "XY4":
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]
    else:
        raise ValueError(f"Unknown DD sequence: {dd_sequence_type}. Use 'XX' or 'XY4'.")
        
    # 4. Schedule and apply PadDynamicalDecoupling
    try:
        pm = PassManager([
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(durations, dd_sequence)
        ])
        dd_qc = pm.run(transpiled_qc)
        if hasattr(transpiled_qc, '_layout'):
            dd_qc._layout = transpiled_qc._layout
        return dd_qc
    except Exception:
        # Fallback if scheduling or DD padding fails (e.g. due to missing durations on simulator)
        return transpiled_qc


def configure_runtime_mitigation_options(
    options, 
    enable_dd: bool = True, 
    dd_sequence: str = "XY4", 
    enable_trex: bool = True
):
    """
    Configures the EstimatorOptions or SamplerOptions with DD + TREX for IBM Quantum Execution.
    
    Args:
        options: EstimatorOptions or SamplerOptions instance.
        enable_dd: Enable Dynamic Decoupling.
        dd_sequence: Sequence type ('XX', 'XpXm', 'XY4').
        enable_trex: Enable Twirled Readout Error eXtinction.
        
    Returns:
        Modified options.
    """
    if enable_dd:
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = dd_sequence
        options.dynamical_decoupling.scheduling_method = "alap"
        
    if enable_trex:
        options.resilience.measure_mitigation = True
        options.twirling.enable_measure = True
        
    return options


def configure_adaptive_mitigation(
    options,
    depth: int,
    two_q_gates: int,
    depth_threshold: int = 80,
    two_q_gate_threshold: int = 100,
    enable_dd: bool = True,
    dd_sequence: str = "XY4",
    enable_trex: bool = True
):
    """
    Configures error mitigation options (DD, TREX, ZNE) adaptively based on circuit depth and 2Q gate count.
    
    Args:
        options: EstimatorOptions (for real/fake IBM backends) or dict (for Aer).
        depth: The transpiled circuit depth (or maximum depth of the sweep).
        two_q_gates: The number of 2-qubit gates (or maximum count of the sweep).
        depth_threshold: Depth threshold below which ZNE is enabled.
        two_q_gate_threshold: 2-qubit gate threshold below which ZNE is enabled.
        enable_dd: If True, enables DD.
        dd_sequence: DD sequence type.
        enable_trex: If True, enables TREX.
    """
    enable_zne = (depth < depth_threshold) and (two_q_gates < two_q_gate_threshold)
    
    is_options_object = hasattr(options, "resilience") or hasattr(options, "dynamical_decoupling")
    
    if is_options_object:
        # DD configuration
        if enable_dd:
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = dd_sequence
            options.dynamical_decoupling.scheduling_method = "alap"
        else:
            options.dynamical_decoupling.enable = False
            
        # TREX configuration
        if enable_trex:
            options.resilience.measure_mitigation = True
            options.twirling.enable_measure = True
        else:
            options.resilience.measure_mitigation = False
            options.twirling.enable_measure = False
            
        # ZNE configuration
        if enable_zne:
            options.resilience.zne_mitigation = True
            options.resilience.zne.extrapolated_noise_factors = [1, 1.2, 1.4]
            # Use PEA if possible or default
            if hasattr(options.resilience.zne, "amplifier"):
                options.resilience.zne.amplifier = "pea"
        else:
            options.resilience.zne_mitigation = False
    else:
        if options is None:
            options = {}
        options["enable_dd"] = enable_dd
        options["dd_sequence"] = dd_sequence
        options["enable_trex"] = enable_trex
        options["enable_zne"] = enable_zne
        
    return options, enable_zne

