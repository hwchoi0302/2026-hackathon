"""
Fetch results of the completed ibm_yonsei batch jobs and generate the final LaTeX + PDF reports.
If the first job is finished but the second job was never submitted (due to laptop shutdown),
this script will automatically submit the second Suzuki-2nd job and track it.
"""
import os, sys, json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonian import get_xyz_hamiltonian
from src.circuit_builder import build_trotter_circuit
from src.execution import get_estimator
from src.analysis import perform_fft_analysis
from legacy_tests.generate_report import main as build_reports

# --- Submit Details ---
LIE_JOB_ID = "d8tablkbp3hs73848log"
SUZUKI_JOB_ID_FILE = "results/suzuki_job_id.txt"

# Model constants
N = 4
JX, JY, JZ, h = 1.0, 0.8, 0.5, 0.5
T_MAX = 2.0
DT = 0.1
OBSERVABLE_STR = "IIIZ"

def get_initial_state(num_qubits: int):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.x(2)
    return qc

def main():
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    print("==========================================================")
    print("IBM Quantum Real Hardware Job Fetch & Resume Suite")
    print("==========================================================")
    
    print("Initializing QiskitRuntimeService...")
    try:
        service = QiskitRuntimeService()
    except Exception as e:
        print("[Error] Failed to load IBM service account. Ensure your token is saved.")
        print(e)
        return

    # 1. Retrieve or check Lie-Trotter job
    print(f"\n1. Querying Lie-Trotter Job (ID: {LIE_JOB_ID})...")
    try:
        lie_job = service.job(LIE_JOB_ID)
        lie_status = lie_job.status().name if hasattr(lie_job.status(), "name") else str(lie_job.status())
        print(f"   Status: {lie_status}")
        
        if lie_status in ["QUEUED", "PENDING", "RUNNING"]:
            print("   [Queue Status] Lie-Trotter is still processing in the IBM Cloud backend.")
            try:
                print(f"   Usage estimation: {lie_job.usage_estimation}")
            except:
                pass
            print("   You can safely turn off your laptop. Run this script again later once the job finishes.")
            return
        elif lie_status not in ["COMPLETED", "DONE"]:
            print(f"   [Warning] Job status is {lie_status}. Job might have failed or been cancelled.")
            return
            
        print("   [Success] Lie-Trotter Job COMPLETED! Downloading results...")
        lie_result = lie_job.result()
    except Exception as e:
        print("[Error] Could not check Lie-Trotter job:", e)
        return

    # 2. Check or Submit Suzuki-Trotter (2nd) Job
    # Check if we have already saved the Suzuki Job ID locally
    suzuki_job_id = None
    if os.path.exists(SUZUKI_JOB_ID_FILE):
        with open(SUZUKI_JOB_ID_FILE, 'r') as f:
            suzuki_job_id = f.read().strip()
            
    # If not saved, scan recent jobs to see if it was submitted by an active harness run
    if not suzuki_job_id:
        print("\n2. Scanning recent jobs to see if Suzuki-2nd Job was already submitted...")
        recent_jobs = service.jobs(limit=10)
        for job in recent_jobs:
            if job.job_id() != LIE_JOB_ID and job.backend().name == "ibm_yonsei":
                # Check tags or creation date
                if job.creation_date > lie_job.creation_date:
                    suzuki_job_id = job.job_id()
                    print(f"   Auto-detected existing Suzuki-2nd Job ID: {suzuki_job_id}")
                    with open(SUZUKI_JOB_ID_FILE, 'w') as f:
                        f.write(suzuki_job_id)
                    break

    # If still not found, it means the harness was cut off (e.g. laptop shutdown) before submitting Suzuki-2nd
    if not suzuki_job_id:
        print("\n2. Suzuki-2nd Job not found. Submitting Suzuki-Trotter (2nd) batch to QPU...")
        try:
            backend_instance = service.backend("ibm_yonsei")
            estimator = get_estimator(
                mode="real", 
                backend_instance=backend_instance,
                enable_dd=True,
                dd_sequence="XY4",
                enable_trex=True
            )
            
            hamiltonian = get_xyz_hamiltonian(N, JX, JY, JZ, h)
            from qiskit.quantum_info import SparsePauliOp
            observable = SparsePauliOp(OBSERVABLE_STR)
            times = np.arange(0.0, T_MAX + DT, DT)
            
            pubs = []
            for t in times:
                if t == 0.0:
                    continue
                steps = max(1, int(np.round(t / 0.2)))
                qc_trotter = build_trotter_circuit(hamiltonian, time=t, steps=steps, model_name="suzuki_2")
                qc = get_initial_state(N).compose(qc_trotter)
                
                from qiskit import transpile
                qc_exec = transpile(qc, backend=backend_instance, optimization_level=2)
                if qc_exec.layout is not None:
                    observable_physical = observable.apply_layout(qc_exec.layout, num_qubits=backend_instance.num_qubits)
                else:
                    observable_physical = observable
                pubs.append((qc_exec, observable_physical))
                
            print("   Submitting Suzuki-2nd batch job to ibm_yonsei...")
            job = estimator.run(pubs, precision=0.01)
            suzuki_job_id = job.job_id()
            print(f"   [Submitted] Job ID: {suzuki_job_id}")
            with open(SUZUKI_JOB_ID_FILE, 'w') as f:
                f.write(suzuki_job_id)
            print("   Suzuki job submitted successfully! You can turn off your laptop now.")
            print("   Run this script again later to fetch the final results once it completes.")
            return
        except Exception as e:
            print("[Error] Failed to submit Suzuki-2nd job:", e)
            return

    # 3. Retrieve Suzuki-Trotter (2nd) results
    print(f"\n3. Querying Suzuki-Trotter (2nd) Job (ID: {suzuki_job_id})...")
    try:
        suz_job = service.job(suzuki_job_id)
        suz_status = suz_job.status().name if hasattr(suz_job.status(), "name") else str(suz_job.status())
        print(f"   Status: {suz_status}")
        
        if suz_status in ["QUEUED", "PENDING", "RUNNING"]:
            print("   [Queue Status] Suzuki-Trotter is still processing in the IBM Cloud backend.")
            print("   You can safely turn off your laptop. Run this script again later once the job finishes.")
            return
        elif suz_status not in ["COMPLETED", "DONE"]:
            print(f"   [Warning] Job status is {suz_status}. Job might have failed.")
            return
            
        print("   [Success] Suzuki-Trotter Job COMPLETED! Downloading results...")
        suz_result = suz_job.result()
    except Exception as e:
        print("[Error] Could not check Suzuki-Trotter job:", e)
        return

    # 4. Reconstruct Expectations & Save
    times = np.arange(0.0, T_MAX + DT, DT)
    active_times = [t for t in times if t != 0.0]
    
    lie_expectations = [-1.0]
    for idx in range(len(active_times)):
        lie_expectations.append(float(lie_result[idx].data.evs))
        
    suz_expectations = [-1.0]
    for idx in range(len(active_times)):
        suz_expectations.append(float(suz_result[idx].data.evs))

    print("\n4. Reconstructed Expectations:")
    print("   Lie-Trotter: ", [round(x, 4) for x in lie_expectations])
    print("   Suzuki-2nd:  ", [round(x, 4) for x in suz_expectations])

    # Run FFT
    print("\n5. Running FFT Spectral Analysis on Real Hardware Data...")
    lie_fft = perform_fft_analysis(times, lie_expectations, window_type="hann", zero_padding_factor=4)
    suz_fft = perform_fft_analysis(times, suz_expectations, window_type="hann", zero_padding_factor=4)

    # Save data to JSON
    real_results = {
        "times": list(times),
        "results": {
            "lie": lie_expectations,
            "suzuki_2": suz_expectations
        },
        "fft_results": {
            "lie": {
                "frequencies": list(lie_fft["frequencies"]),
                "spectrum": list(lie_fft["spectrum"]),
                "peak_frequencies": list(lie_fft["peak_frequencies"]),
                "peak_values": list(lie_fft["peak_values"])
            },
            "suzuki_2": {
                "frequencies": list(suz_fft["frequencies"]),
                "spectrum": list(suz_fft["spectrum"]),
                "peak_frequencies": list(suz_fft["peak_frequencies"]),
                "peak_values": list(suz_fft["peak_values"])
            }
        }
    }

    os.makedirs('results', exist_ok=True)
    json_path = "results/harness_real_results.json"
    with open(json_path, 'w') as f:
        json.dump(real_results, f, indent=4, default=str)
    print(f"   Saved retrieved real results to: {json_path}")

    # Regenerate plots and compile LaTeX
    print("\n6. Regenerating plots and LaTeX + PDF reports...")
    build_reports()
    
    # Compile PDF
    os.system("pdflatex -output-directory=results results/simulation_report.tex")
    print("\n[✔] Final real hardware report generated and compiled successfully!")

if __name__ == "__main__":
    main()
