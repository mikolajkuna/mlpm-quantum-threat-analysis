"""
experiment2_grover_quantum_inspire.py
======================================
Grover's algorithm demonstration on Quantum Inspire (Paper 3.5, Experiment 2).

Constructs a 3-qubit Grover circuit searching for marked bypass state |101⟩
(MARKED=5) in N=8 state space. Sweeps k=0..MAX_ITER Grover iterations and
compares empirical vs theoretical success probability.

Setup (one-time):
    python -m pip install qiskit qiskit-aer qiskit-quantuminspire
    qi login

Usage:
    python experiment2_grover_quantum_inspire.py --backend local    # Qiskit Aer
    python experiment2_grover_quantum_inspire.py --backend qi_sim   # QX Emulator
    python experiment2_grover_quantum_inspire.py --backend qi_hw    # QI hardware

Expected results (k*=2):
    P_empirical  = 0.940
    P_theoretical = 0.945
    Error: 0.53%

Outputs:
    results/experiment2/grover_{backend}_{timestamp}.json
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("../results/experiment2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS  = 3
N_STATES  = 2 ** N_QUBITS   # 8
MARKED    = 5                # |101⟩
SHOTS     = 1024
MAX_ITER  = 3                # sweep k = 0 .. MAX_ITER

QI_BACKEND_SIM = "QX emulator"
QI_BACKEND_HW  = "Tuna-5"    # Available: Tuna-5, Tuna-9, Tuna-17


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------

def theoretical_grover_prob(k: int, n_states: int, n_marked: int = 1) -> float:
    """P_success = sin^2((2k+1) * theta), theta = arcsin(sqrt(m/N))."""
    theta = math.asin(math.sqrt(n_marked / n_states))
    return math.sin((2 * k + 1) * theta) ** 2


def theoretical_classical_prob(k: int, n_states: int) -> float:
    """Expected classical success probability after k oracle queries."""
    return min(k / n_states, 1.0)


def optimal_iterations(n_states: int, n_marked: int = 1) -> int:
    """k* = floor(pi/4 * sqrt(N/m))."""
    return math.floor(math.pi / 4 * math.sqrt(n_states / n_marked))


# ---------------------------------------------------------------------------
# Grover circuit construction (Qiskit)
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked: int):
    """
    Phase oracle: flips phase of |marked⟩.

    Uses Qiskit native mcx gate. Pre-decomposed to {cx, h, x, t, tdg, s, sdg}
    basis before submission to QI cQASM backend.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name="Oracle")
    bits = format(marked, f"0{n_qubits}b")

    # X gates on qubits where bit is 0 (to make |marked⟩ → |11...1⟩)
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)

    # Multi-controlled phase flip via H + MCX + H on target qubit
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    # Undo X gates
    for i, bit in enumerate(reversed(bits)):
        if bit == "0":
            qc.x(i)

    return qc


def build_diffuser(n_qubits: int):
    """
    Grover diffusion operator: 2|s⟩⟨s| - I.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked: int, k: int):
    """
    Full Grover circuit for k iterations.
    Returns QuantumCircuit with measurement.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    # k Grover iterations
    oracle   = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    for _ in range(k):
        qc.compose(oracle,   inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def decompose_for_qi(circuit):
    """
    Transpile to basis gates supported by QI cQASM backend:
    {cx, h, x, t, tdg, s, sdg}.

    This resolves mcx (Toffoli) incompatibility with QI hardware.
    """
    from qiskit import transpile

    return transpile(
        circuit,
        basis_gates=["cx", "h", "x", "t", "tdg", "s", "sdg", "measure"],
        optimization_level=1,
    )


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

def get_backend_local():
    from qiskit_aer import AerSimulator
    return AerSimulator(), "local_aer"


def get_backend_qi_sim():
    from quantuminspire.sdk.qiskit import QIProvider
    provider = QIProvider()
    backend  = provider.get_backend(QI_BACKEND_SIM)
    return backend, f"qi_{QI_BACKEND_SIM.replace(' ', '_')}"


def get_backend_qi_hw():
    from quantuminspire.sdk.qiskit import QIProvider
    provider = QIProvider()
    backend  = provider.get_backend(QI_BACKEND_HW)
    return backend, f"qi_{QI_BACKEND_HW}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_circuits(circuits: list, backend, shots: int) -> list[dict]:
    """Run list of circuits and return counts for each."""
    from qiskit import transpile

    counts_list = []
    for circuit in circuits:
        job    = backend.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        counts_list.append(counts)
    return counts_list


def extract_success_prob(counts: dict, marked: int, n_qubits: int, shots: int) -> float:
    """Empirical probability of measuring the marked state."""
    marked_str  = format(marked, f"0{n_qubits}b")
    marked_count = counts.get(marked_str, 0)
    return marked_count / shots


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: dict):
    print(f"\n{'k':>4} {'P_emp':>10} {'P_theor':>10} {'P_class':>10} {'Error%':>8}")
    print("-" * 46)
    for k, row in sorted(results.items()):
        err = f"{row['error_vs_theory_pct']:.2f}%" if row["error_vs_theory_pct"] is not None else "—"
        print(
            f"{k:>4} {row['p_success_empirical']:>10.4f} "
            f"{row['p_success_theoretical']:>10.4f} "
            f"{row['p_classical']:>10.4f} "
            f"{err:>8}"
        )
    k_opt = optimal_iterations(N_STATES)
    print(f"\nOptimal k* = {k_opt}  (π/4·√{N_STATES} ≈ {math.pi/4*math.sqrt(N_STATES):.3f})")


def save_results(results: dict, backend_label: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"grover_{backend_label}_{timestamp}.json"

    payload = {
        "config": {
            "n_qubits":  N_QUBITS,
            "n_states":  N_STATES,
            "marked":    MARKED,
            "marked_str": format(MARKED, f"0{N_QUBITS}b"),
            "shots":     SHOTS,
            "max_iter":  MAX_ITER,
            "backend":   backend_label,
            "k_optimal": optimal_iterations(N_STATES),
        },
        "results": {str(k): v for k, v in results.items()},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grover experiment on Quantum Inspire")
    parser.add_argument(
        "--backend",
        choices=["local", "qi_sim", "qi_hw"],
        default="local",
        help="Backend: local (Aer), qi_sim (QX Emulator), qi_hw (Tuna hardware)",
    )
    parser.add_argument("--shots", type=int, default=SHOTS)
    args = parser.parse_args()

    shots = args.shots

    # Select backend
    if args.backend == "local":
        backend, label = get_backend_local()
        decompose = False
    elif args.backend == "qi_sim":
        backend, label = get_backend_qi_sim()
        decompose = True
    else:
        backend, label = get_backend_qi_hw()
        decompose = True

    print(f"Backend: {label}  |  shots={shots}")
    print(f"Marked state: |{format(MARKED, f'0{N_QUBITS}b')}⟩ = |{MARKED}⟩  "
          f"in N={N_STATES} states")

    # Build circuits for k = 0 .. MAX_ITER
    circuits = []
    for k in range(MAX_ITER + 1):
        qc = build_grover_circuit(N_QUBITS, MARKED, k)
        if decompose:
            qc = decompose_for_qi(qc)
        circuits.append(qc)

    print(f"\nRunning {len(circuits)} circuits (k=0..{MAX_ITER})...")

    all_counts = run_circuits(circuits, backend, shots)

    # Collect results
    results = {}
    for k, counts in enumerate(all_counts):
        p_emp = extract_success_prob(counts, MARKED, N_QUBITS, shots)
        p_th  = theoretical_grover_prob(k, N_STATES)
        p_cl  = theoretical_classical_prob(k, N_STATES)
        err   = (abs(p_emp - p_th) / p_th * 100) if p_th > 0 else None

        results[k] = {
            "k":                     k,
            "counts":                counts,
            "p_success_empirical":   round(p_emp, 4),
            "p_success_theoretical": round(p_th,  4),
            "p_classical":           round(p_cl,  4),
            "error_vs_theory_pct":   round(err, 2) if err is not None else None,
        }

    print_table(results)
    save_results(results, label)


if __name__ == "__main__":
    main()
