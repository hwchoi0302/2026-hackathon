import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from qiskit import QuantumCircuit, transpile
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

# ── Fixed Infidelity Step Mapping (Epsilon = 0.05) ──────────────────────────────
STEP_MAPPING = {
    "lie":      [0, 1, 1, 1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 15, 16, 17, 19, 20, 20, 21, 21],
    "suzuki_2": [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,  3,  3,  3,  3,  4,  4,  4,  5,  5,  5],
    "suzuki_4": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2]
}

def get_initial_state(num_qubits: int) -> QuantumCircuit:
    """Prepares the initial Neel state |1010>."""
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.x(2)
    return qc

def get_exact_expectations(times):
    """Computes or loads exact Z0 expectations."""
    exact_path = "results/exact_diagonalization.json"
    if os.path.exists(exact_path):
        try:
            with open(exact_path, 'r') as f:
                data = json.load(f)
                return np.array(data["exact_Z0"]), data
        except Exception:
            pass
            
    # Attempt to load from non-qkd1 fallback
    fallback_path = "results/non-qkd1/exact_diagonalization.json"
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, 'r') as f:
                data = json.load(f)
                # Write to main results for future use
                with open(exact_path, 'w') as out_f:
                    json.dump(data, out_f, indent=4)
                return np.array(data["exact_Z0"]), data
        except Exception:
            pass
            
    # Fallback to computing exact expectations on the fly
    print("Exact diagonalization file not found. Computing on the fly...")
    from scipy.linalg import expm
    hamiltonian = get_xyz_hamiltonian(N, JX, JY, JZ, h)
    H_matrix = hamiltonian.to_matrix()
    Z0 = SparsePauliOp(OBSERVABLE_STR).to_matrix()
    
    # Neel state
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[5] = 1.0 # q3 q2 q1 q0 = 0 1 0 1 -> decimal 5
    
    exact_vals = []
    for t in times:
        if t == 0.0:
            val = np.real(psi0.conj() @ Z0 @ psi0)
        else:
            U = expm(-1j * H_matrix * t)
            psi_t = U @ psi0
            val = np.real(psi_t.conj() @ Z0 @ psi_t)
        exact_vals.append(float(val))
        
    # Construct exact spectrum data
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    energy_diffs = []
    for i in range(len(eigenvalues)):
        for j in range(i+1, len(eigenvalues)):
            energy_diffs.append(eigenvalues[j] - eigenvalues[i])
    energy_diffs = sorted(list(set(np.round(energy_diffs, 8))))
    overlaps = [float(np.abs(eigenvectors[:, i].conj() @ psi0)**2) for i in range(len(eigenvalues))]
    
    data = {
        "eigenvalues": [float(e) for e in eigenvalues],
        "energy_diffs": [float(d) for d in energy_diffs],
        "frequencies_hz": [float(d/(2*np.pi)) for d in energy_diffs],
        "ground_state_energy": float(eigenvalues[0]),
        "overlaps": overlaps,
        "times": list(times),
        "exact_Z0": exact_vals
    }
    with open(exact_path, 'w') as f:
        json.dump(data, f, indent=4)
    return np.array(exact_vals), data

def main():
    parser = argparse.ArgumentParser(description="1D XYZ Trotterization non-qkd2 Harness")
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

    os.makedirs('results/non-qkd2', exist_ok=True)
    print("=" * 70)
    print(f"Executing non-qkd2 XYZ Trotterization Harness: {args.backend.upper()}")
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

    # 2. Setup times and load exact expectations
    times = np.arange(0.0, T_MAX + DT, DT)
    exact_Z0, exact_data = get_exact_expectations(times)

    # Models to compare
    models = [m.strip() for m in args.models.split(",")]
    # For real hardware, skip suzuki_4 if requested
    if args.backend == "real" and "suzuki_4" in models:
        print("  [Warning] Suzuki-4th is skipped for real hardware execution.")
        models = [m for m in models if m != "suzuki_4"]
        
    model_labels = {
        "lie": "1st-order Lie-Trotter",
        "suzuki_2": "2nd-order Suzuki-Trotter",
        "suzuki_4": "4th-order Suzuki-Trotter"
    }

    # 3. Build XYZ Hamiltonian
    hamiltonian = get_xyz_hamiltonian(N, JX, JY, JZ, h)
    observable = SparsePauliOp(OBSERVABLE_STR)

    all_results = {}

    # 4. Sweep execution over time points
    for model in models:
        print(f"\nPreparing circuits for {model_labels[model]}...")
        
        # Determine maximum circuit depth and set resilience level
        # Lie-Trotter: max r=21, 2Q gates = 252. Suzuki-2nd: max r=5, 2Q gates = 120. Suzuki-4th: max r=2, 2Q gates = 240.
        r_list = STEP_MAPPING[model]
        max_r = max(r_list)
        # Factor of 2Q gates per step (Lie: 12, Suzuki-2: 24, Suzuki-4: 120)
        factor = 12 if model == "lie" else (24 if model == "suzuki_2" else 120)
        max_2q_gates = max_r * factor
        
        # Decide ZNE based on gate depth
        if max_2q_gates < 150:
            res_level = 2  # Enable ZNE
            zne_status = "ENABLED (resilience_level=2)"
        else:
            res_level = 1  # Disable ZNE
            zne_status = "DISABLED (resilience_level=1)"
            
        print(f"  Max Steps: {max_r} | Max 2Q Gates: {max_2q_gates} | ZNE Status: {zne_status}")

        # Get EstimatorV2 with correct resilience level
        estimator = get_estimator(
            mode=args.backend, 
            backend_instance=backend_instance,
            enable_dd=args.enable_dd,
            dd_sequence=args.dd_sequence,
            enable_trex=args.enable_trex,
            resilience_level=res_level
        )

        pubs = []
        active_times = []
        
        for idx, t in enumerate(times):
            if t == 0.0:
                continue
                
            active_times.append(t)
            steps = STEP_MAPPING[model][idx]
            
            # Build Trotter circuit
            qc_trotter = build_trotter_circuit(hamiltonian, time=t, steps=steps, model_name=model)
            qc = get_initial_state(N).compose(qc_trotter)
            
            # Apply local DD or transpile
            if args.backend in ["noiseless", "fake"] and args.enable_dd:
                qc_exec = apply_local_dd(qc, backend_instance, dd_sequence_type=args.dd_sequence)
            else:
                qc_exec = transpile(qc, backend=backend_instance, optimization_level=2)
                
            # Map observable to physical layout
            if qc_exec.layout is not None:
                observable_physical = observable.apply_layout(qc_exec.layout, num_qubits=backend_instance.num_qubits)
            else:
                observable_physical = observable
                
            pubs.append((qc_exec, observable_physical))
            
            depth_exec = qc_exec.depth()
            two_q_exec = qc_exec.num_nonlocal_gates()
            print(f"  t={t:.2f} (r={steps}) | Depth: {depth_exec:3d} | 2Q Gates: {two_q_exec:3d} (prepared)")

        # Run batch
        print(f"Submitting batch job for {model_labels[model]} to the estimator backend...")
        job = estimator.run(pubs, precision=0.01)
        
        if hasattr(job, "job_id"):
            print(f"  Submitted Job ID: {job.job_id()}")
            print("  Waiting for execution to finish...")
            
        result = job.result()
        
        # Reconstruct the expectations list (t=0.0 is -1.0)
        expectations = [-1.0]
        for idx_t, t in enumerate(active_times):
            pub_result = result[idx_t]
            val = float(pub_result.data.evs)
            expectations.append(val)
            print(f"  Result t={t:.2f} | <Z_0>: {val:.4f}")
            
        all_results[model] = np.array(expectations)

    # 5. FFT Analysis
    print("\n" + "-"*50)
    print("Decoupled FFT Spectral Analysis Results (non-qkd2):")
    print("-"*50)
    
    fft_results = {}
    for model in models:
        res = perform_fft_analysis(times, all_results[model], window_type="hann", zero_padding_factor=4)
        fft_results[model] = res
        print(f"Model: {model_labels[model]}")
        print(f"  Peak Frequencies: {res['peak_frequencies'][:3].round(3)}")
        print(f"  Peak Magnitudes:  {res['peak_values'][:3].round(3)}")
        print()

    # 6. Plotting (including Exact value comparison)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: Time-domain expectation values vs Exact
    ax = axes[0]
    colors = {"lie": "royalblue", "suzuki_2": "tomato", "suzuki_4": "mediumseagreen"}
    markers = {"lie": "o", "suzuki_2": "s", "suzuki_4": "^"}
    
    # Plot Exact value as a black dashed line
    ax.plot(times, exact_Z0, label="Exact (Theory)", color="black", linestyle="--", linewidth=2.5, zorder=5)
    
    for model in models:
        ax.plot(
            times, 
            all_results[model], 
            label=model_labels[model], 
            color=colors[model], 
            marker=markers[model],
            linestyle="-",
            zorder=4
        )
    ax.set_xlabel("Time ($t$)", fontsize=12)
    ax.set_ylabel(r"$\langle Z_0 \rangle$", fontsize=12)
    ax.set_title(f"XYZ Simulation Z0 Expectations vs Exact ({args.backend.upper()})", fontsize=12, fontweight="bold")
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
    plot_path = f"results/non-qkd2/harness_{args.backend}_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to: {plot_path}")

    # Compute fidelity metrics
    print("\n" + "-"*50)
    print("Fidelity & Error Metrics vs Exact (non-qkd2):")
    print("-"*50)
    
    backend_label = {
        "noiseless": "Noiseless",
        "fake": "Fake",
        "real": "Real"
    }[args.backend]
    
    new_metrics = {}
    for model in models:
        sim_v = all_results[model]
        rms = np.sqrt(np.mean((sim_v - exact_Z0)**2))
        max_err = np.max(np.abs(sim_v - exact_Z0))
        corr = np.corrcoef(exact_Z0, sim_v)[0, 1] if np.std(sim_v) > 1e-10 else 0.0
        
        new_metrics[model] = {
            "fidelity": float(corr),
            "rmse": float(rms),
            "max_abs_error": float(max_err)
        }
        
        print(f"{model_labels[model]}:")
        print(f"  Fidelity (Corr): {corr:.6f}")
        print(f"  RMSE:            {rms:.6f}")
        print(f"  Max Absolute Er: {max_err:.6f}")

    # Update final_summary.json
    summary_path = "results/non-qkd2/final_summary.json"
    non_zero_diffs = [d for d in exact_data["energy_diffs"] if d > 1e-6]
    first_gap = float(non_zero_diffs[0]) if non_zero_diffs else 0.0
    summary_data = {
        "exact_ground_energy": exact_data["ground_state_energy"],
        "exact_first_gap": first_gap,
        "metrics": {},
        "fft_peaks": {}
    }
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
        except Exception:
            pass
            
    # Always ensure exact ground truth values are correctly written
    summary_data["exact_ground_energy"] = exact_data["ground_state_energy"]
    summary_data["exact_first_gap"] = first_gap
            
    # Update metrics for current backend
    for model in models:
        key = f"{backend_label} {model.capitalize()}"
        if model == "suzuki_2":
            key = f"{backend_label} Suzuki-2"
        elif model == "suzuki_4":
            key = f"{backend_label} Suzuki-4"
            
        summary_data["metrics"][key] = new_metrics[model]
        
        # Save first two peak frequencies
        peak_freqs = list(fft_results[model]["peak_frequencies"][:2])
        summary_data["fft_peaks"][f"{args.backend}_{model}"] = [float(f) for f in peak_freqs]
        
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Updated final summary metrics in: {summary_path}")

    # Save detailed data to JSON
    json_data = {
        "times": list(times),
        "exact_Z0": list(exact_Z0),
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
    json_path = f"results/non-qkd2/harness_{args.backend}_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4, default=str)
    print(f"Saved results data to: {json_path}")
    print("\nnon-qkd2 Harness run completed successfully!")

if __name__ == "__main__":
    main()
