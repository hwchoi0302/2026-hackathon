# Quantum Simulation of 1D Periodic Heisenberg Model

This repository contains the implementation and experimentation scripts for simulating the 1D Periodic Heisenberg Model using Trotterization on IBM Quantum hardware.

## Directory Structure

- `src/`: Core physics and simulation modules.
  - `heisenberg.py`: Functions to generate the Hamiltonian, exact time evolution, and 1st/2nd order Trotter circuits.
  - `error_mitigation.py`: Implementation of Zero Noise Extrapolation (ZNE) using Mitiq.
  - `spectral_analysis.py`: Tools for frequency domain analysis of the quantum dynamics.
- `scripts/`: Executable scripts for running experiments.
  - `debug_trotter.py`: Verifies the correctness of the Trotter circuits against the exact mathematical evolution.
  - `run_experiments.py`: Runs noisy simulation experiments on `FakeBrisbane` with ZNE.
  - `run_real_hardware.py`: Script to submit the Trotterized circuits to an actual IBM Quantum backend.
  - `run_opt_comparison.py`: Compares different Qiskit optimization levels (1 vs 3) for the Trotter circuits.
  - `run_r_sweep.py`: Sweeps Trotter steps `r` from 1 to 5 to observe the trade-off between Trotter error and real hardware noise.

## Setup and Requirements

1. **Virtual Environment**:
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Dependencies**:
   Install the required quantum computing libraries:
   ```bash
   pip install qiskit qiskit-ibm-runtime qiskit-aer mitiq numpy scipy matplotlib
   ```

## Usage Instructions

### 1. Verification of Circuits
To ensure that the 1st-order and 2nd-order Trotter circuits correctly approximate the Heisenberg Hamiltonian:
```bash
python scripts/debug_trotter.py
```

### 2. Noisy Simulation (Mock Backend)
To run the ZNE experiment on the `FakeBrisbane` mock backend:
```bash
python scripts/run_experiments.py
```
This will output exact values, unmitigated noisy values, and ZNE (Richardson) mitigated values.

### 3. Real Hardware Execution
To execute on a real IBM Quantum backend (e.g., `ibm_yonsei`), you must first provide your credentials:

```bash
# Set your IBM Cloud IAM API Key and CRN
export IBM_IAM_APIKEY="your-api-key"
export IBM_CRN="your-instance-crn"

python scripts/run_real_hardware.py
```

### 4. Hardware vs. Trotter Error Trade-off Analysis
To analyze the sweet spot between theoretical Trotter error and physical hardware noise:
```bash
python scripts/run_r_sweep.py
```
This sweeps $r \in [1, 5]$ using optimization level 3 and compares the theoretical curve to the actual hardware execution.
