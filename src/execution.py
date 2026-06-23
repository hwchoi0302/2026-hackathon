from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.options import EstimatorOptions
from src.mitigation import configure_runtime_mitigation_options

def get_estimator(mode: str, backend_instance=None, enable_dd: bool = True, dd_sequence: str = "XY4", enable_trex: bool = True):
    """
    Returns the appropriate EstimatorV2 instance based on the execution mode.
    
    Args:
        mode: One of 'noiseless', 'fake', 'real'.
        backend_instance: The backend instance (required for 'real', optional for 'fake').
        enable_dd: If True, enables Dynamic Decoupling (for 'real' mode).
        dd_sequence: DD sequence type ('XX', 'XY4').
        enable_trex: If True, enables twirled readout error mitigation (for 'real' mode).
        
    Returns:
        An EstimatorV2 instance.
    """
    if mode == 'noiseless':
        return AerEstimator()
        
    elif mode == 'fake':
        if backend_instance is None:
            from qiskit_ibm_runtime.fake_provider import FakeBrisbane
            backend_instance = FakeBrisbane()
            
        noise_model = NoiseModel.from_backend(backend_instance)
        options = {
            'backend_options': {
                'noise_model': noise_model,
                'basis_gates': backend_instance.operation_names,
                'coupling_map': backend_instance.coupling_map
            }
        }
        return AerEstimator(options=options)
        
    elif mode == 'real':
        if backend_instance is None:
            raise ValueError("backend_instance is required for 'real' execution mode.")
            
        options = EstimatorOptions()
        # Set shots or other options if needed, then configure DD/TREX
        configure_runtime_mitigation_options(
            options, 
            enable_dd=enable_dd, 
            dd_sequence=dd_sequence, 
            enable_trex=enable_trex
        )
        return IBMEstimator(mode=backend_instance, options=options)
        
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
