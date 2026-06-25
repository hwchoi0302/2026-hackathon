from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap

def get_xyz_hamiltonian(N: int, Jx: float, Jy: float, Jz: float, h: float) -> SparsePauliOp:
    """
    Constructs the 1D anisotropic XYZ Hamiltonian as a SparsePauliOp.
    
    H = sum_{<i,j>} (Jx * X_i X_j + Jy * Y_i Y_j + Jz * Z_i Z_j) + h * sum_{i} Z_i
    
    using a ring coupling map for periodic boundary conditions.
    
    Args:
        N: Number of qubits.
        Jx: XX coupling strength.
        Jy: YY coupling strength.
        Jz: ZZ coupling strength.
        h: Transverse field strength (Z).
                  
    Returns:
        SparsePauliOp representing the XYZ Hamiltonian.
    """
    if N > 2:
        coupling_map = CouplingMap.from_ring(N)
    else:
        coupling_map = CouplingMap.from_line(N)
        
    # Extract unique undirected bonds
    bonds = set()
    for u, v in coupling_map.get_edges():
        bonds.add(tuple(sorted((u, v))))
        
    terms = []
    
    # 2-qubit interactions: Jx * XX, Jy * YY, Jz * ZZ
    for u, v in sorted(list(bonds)):
        # XX term
        pauli_list_x = ['I'] * N
        pauli_list_x[u] = 'X'
        pauli_list_x[v] = 'X'
        terms.append(("".join(pauli_list_x[::-1]), Jx))
        
        # YY term
        pauli_list_y = ['I'] * N
        pauli_list_y[u] = 'Y'
        pauli_list_y[v] = 'Y'
        terms.append(("".join(pauli_list_y[::-1]), Jy))
        
        # ZZ term
        pauli_list_z = ['I'] * N
        pauli_list_z[u] = 'Z'
        pauli_list_z[v] = 'Z'
        terms.append(("".join(pauli_list_z[::-1]), Jz))
            
    # 1-qubit field: Z
    for i in range(N):
        pauli_list = ['I'] * N
        pauli_list[i] = 'Z'
        terms.append(("".join(pauli_list[::-1]), h))
        
    return SparsePauliOp.from_list(terms)
