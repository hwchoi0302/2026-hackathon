import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian
from src.circuit_builder import build_trotter_circuit
from src.mitigation import apply_local_dd
from src.execution import get_estimator
from src.analysis import perform_fft_analysis

# ── XYZ Hamiltonian Parameters ──────────────────────────────────────────────────
N = 4
JX = 1.0
JY = 0.8
JZ = 0.5
h = 0.5

T_MAX = 2.0
DT = 0.1  # Time step resolution
OBSERVABLE_STR = "IIIZ"  # Z expectation on qubit 0 (little-endian)

def get_initial_state(num_qubits: int) -> QuantumCircuit:
    """Prepares the initial Neel state |1010>."""
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.x(2)
    return qc

def main():
    parser = argparse.ArgumentParser(description="1D XYZ Trotterization Execution Harness")
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["noiseless", "fake", "real"], 
        default="noiseless",
        help="Backend to execute on: noiseless simulator, noisy FakeBrisbane, or real hardware."
    )
    parser.add_argument(
        "--enable-dd", 
        action="store_true", 
        default=True,
        help="Enable Dynamic Decoupling (DD) error mitigation."
    )
    parser.add_argument(
        "--dd-sequence", 
        type=str, 
        default="XY4", 
        choices=["XX", "XY4"],
        help="Dynamic decoupling sequence to use."
    )
    parser.add_argument(
        "--enable-trex", 
        action="store_true", 
        default=True,
        help="Enable Twirled Readout Error eXtinction (TREX) (real hardware only)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lie,suzuki_2,suzuki_4",
        help="Comma-separated list of models to run: lie, suzuki_2, suzuki_4"
    )
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    print("=" * 70)
    print(f"Executing XYZ Trotterization Harness: {args.backend.upper()}")
    print("=" * 70)

    # 1. Setup Backend
    backend_instance = None
    if args.backend == "noiseless":
        backend_instance = AerSimulator()
        print("  Using Noiseless AerSimulator")
    elif args.backend == "fake":
        backend_instance = FakeBrisbane()
        print(f"  Using Noisy FakeBrisbane ({backend_instance.num_qubits} qubits)")
    elif args.backend == "real":
        from qiskit_ibm_runtime import QiskitRuntimeService
        iam_key = os.environ.get("IBM_IAM_APIKEY", "")
        crn = os.environ.get("IBM_CRN", "")
        if iam_key and crn:
            service = QiskitRuntimeService(channel="ibm_cloud", token=iam_key, instance=crn)
        else:
            print("  No credentials in environment variables. Loading saved account...")
            service = QiskitRuntimeService()
        backend_instance = service.backend("ibm_yonsei")
        print(f"  Connected to real hardware: {backend_instance.name} ({backend_instance.num_qubits} qubits)")

    # 2. Get EstimatorV2
    estimator = get_estimator(
        mode=args.backend, 
        backend_instance=backend_instance,
        enable_dd=args.enable_dd,
        dd_sequence=args.dd_sequence,
        enable_trex=args.enable_trex
    )

    # 3. Build XYZ Hamiltonian
    hamiltonian = get_xyz_hamiltonian(N, JX, JY, JZ, h)
    observable = SparsePauliOp(OBSERVABLE_STR)
    
    # Time points to sweep
    times = np.arange(0.0, T_MAX + DT, DT)
    
    # Models to compare
    models = [m.strip() for m in args.models.split(",")]
    model_labels = {
        "lie": "1st-order Lie-Trotter",
        "suzuki_2": "2nd-order Suzuki-Trotter",
        "suzuki_4": "4th-order Suzuki-Trotter"
    }

    all_results = {}

    # 4. Sweep execution over time points using single-batch execution
    for model in models:
        print(f"\nPreparing circuits for {model_labels[model]}...")
        
        # We will collect all circuits to run in batch
        pubs = []
        active_times = []
        
        for t in times:
            if t == 0.0:
                continue
                
            active_times.append(t)
            # Compute steps (keep dt = 0.2 constant)
            steps = max(1, int(np.round(t / 0.2)))
            
            # Build Trotter circuit
            qc_trotter = build_trotter_circuit(hamiltonian, time=t, steps=steps, model_name=model)
            qc = get_initial_state(N).compose(qc_trotter)
            
            # Apply local DD or transpile
            if args.backend in ["noiseless", "fake"] and args.enable_dd:
                qc_exec = apply_local_dd(qc, backend_instance, dd_sequence_type=args.dd_sequence)
            else:
                from qiskit import transpile
                qc_exec = transpile(qc, backend=backend_instance, optimization_level=2)
                
            # Map observable to physical layout if applicable
            if qc_exec.layout is not None:
                observable_physical = observable.apply_layout(qc_exec.layout, num_qubits=backend_instance.num_qubits)
            else:
                observable_physical = observable
                
            pubs.append((qc_exec, observable_physical))
            
            # Print post-transpilation metrics
            depth_exec = qc_exec.depth()
            two_q_exec = qc_exec.num_nonlocal_gates()
            print(f"  t={t:.2f} (r={steps}) | Depth: {depth_exec:3d} | 2Q Gates: {two_q_exec:3d} (prepared)")

        # Run all collected points in a single job
        print(f"Submitting batch job for {model_labels[model]} to the estimator backend...")
        job = estimator.run(pubs, precision=0.01)
        
        # For real backends, print details immediately so user can monitor via IBM Quantum console
        if hasattr(job, "job_id"):
            print(f"  Submitted Job ID: {job.job_id()}")
            print("  Waiting for execution to finish (this will block until the job completes)...")
            
        result = job.result()
        
        # Reconstruct the expectation values list (t=0.0 is -1.0)
        expectations = [-1.0]
        for idx, t in enumerate(active_times):
            pub_result = result[idx]
            val = float(pub_result.data.evs)
            expectations.append(val)
            print(f"  Result t={t:.2f} | <Z_0>: {val:.4f}")
            
        all_results[model] = np.array(expectations)

    # 5. Decoupled User FFT Analysis
    print("\n" + "-"*50)
    print("Decoupled FFT Spectral Analysis Results:")
    print("-"*50)
    
    fft_results = {}
    for model in models:
        res = perform_fft_analysis(times, all_results[model], window_type="hann", zero_padding_factor=4)
        fft_results[model] = res
        print(f"Model: {model_labels[model]}")
        print(f"  Peak Frequencies: {res['peak_frequencies'][:3].round(3)}")
        print(f"  Peak Magnitudes:  {res['peak_values'][:3].round(3)}")
        print()

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: Time-domain expectation values
    ax = axes[0]
    colors = {"lie": "royalblue", "suzuki_2": "tomato", "suzuki_4": "mediumseagreen"}
    markers = {"lie": "o", "suzuki_2": "s", "suzuki_4": "^"}
    
    for model in models:
        ax.plot(
            times, 
            all_results[model], 
            label=model_labels[model], 
            color=colors[model], 
            marker=markers[model],
            linestyle="-"
        )
    ax.set_xlabel("Time ($t$)", fontsize=12)
    ax.set_ylabel(r"$\langle Z_0 \rangle$", fontsize=12)
    ax.set_title(f"XYZ Simulation expectation values ({args.backend.upper()})", fontsize=12, fontweight="bold")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)
    ax.legend()

    # Right subplot: Frequency-domain FFT spectrum
    ax = axes[1]
    for model in models:
        ax.plot(
            fft_results[model]["frequencies"], 
            fft_results[model]["spectrum"], 
            label=model_labels[model], 
            color=colors[model],
            linewidth=2
        )
    ax.set_xlabel("Frequency ($f$)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.set_title("Spectral Analysis via FFT", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = f"results/harness_{args.backend}_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {plot_path}")

    # Save data to JSON
    json_data = {
        "times": list(times),
        "results": {model: list(all_results[model]) for model in models},
        "fft_results": {
            model: {
                "frequencies": list(fft_results[model]["frequencies"]),
                "spectrum": list(fft_results[model]["spectrum"]),
                "peak_frequencies": list(fft_results[model]["peak_frequencies"]),
                "peak_values": list(fft_results[model]["peak_values"])
            } for model in models
        }
    }
    json_path = f"results/harness_{args.backend}_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4, default=str)
    print(f"Saved results data to: {json_path}")
    print("\nHarness run completed successfully!")

if __name__ == "__main__":
    main()
