# mlpm-quantum-threat-analysis

**Kwantowa analiza zagrożeń adwersarialnych dla prototypowych systemów moderacji małych modeli językowych**
*Quantum Adversarial Threat Analysis for Prototype-Based Moderation of Small Language Models*

---

## Overview

This repository contains the experimental code for the paper co-authored by Mikołaj Kuna and Marcin Kowalczyk (Faculty of Electronics and Information Technology, Warsaw University of Technology).

The paper presents a quantum adversarial complexity analysis of the MLPM (Latent Prototype Moderator) mechanism applied to on-premise SLM deployments in HR pay equity compliance systems.

The dataset will be released publicly upon paper acceptance at conference. Place `hr_payequity_adv.json` in this directory before running experiments.

### Main contributions

1. **Calibration Paradox** — formal derivation showing that calibration reducing bypass count by factor α increases classical attack difficulty by α but quantum (Grover) difficulty by only √α, amplifying the quantum attacker's relative advantage.

2. **Experiment 1** — empirical verification of the Calibration Paradox on the HR-PayEquity-Adv dataset across four SLM models (threshold sweep, bypass density ρ as a function of τ).

3. **Experiment 2** — Grover's algorithm demonstration on the Quantum Inspire QX Emulator (n=3 qubits, N=8 states, marked state |101⟩, k*=2 optimal iterations).

---

## Models evaluated

| Model | HuggingFace ID |
|---|---|
| Llama 3.1 8B Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Phi-3.5 Mini Instruct | `microsoft/Phi-3.5-mini-instruct` |
| IBM Granite 3.3 8B Instruct | `ibm-granite/granite-3.3-8b-instruct` |

---

## Repository structure

```
mlpm-quantum-threat-analysis/
├── experiments/
│   ├── eval_mlpm_with_scores.py           # MLPM inference with per-prompt scores
│   ├── experiment1_calibration_paradox.py # Empirical Calibration Paradox verification
│   └── experiment2_grover_quantum_inspire.py # Grover demo on Quantum Inspire
├── results/
│   ├── experiment1/                       # JSON/CSV outputs from Exp. 1 (local)
│   └── experiment2/                       # JSON outputs from Exp. 2 (local)
├── data/
│   └── README.md                          # HR-PayEquity-Adv dataset description
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.13 (tested with venv313)
- CUDA-capable GPU recommended for Experiment 1 (tested: RTX 5060 8GB)
- Quantum Inspire account for Experiment 2 (free tier sufficient)

### Installation

```bash
git clone https://github.com/mikolajkuna/mlpm-quantum-threat-analysis.git
cd mlpm-quantum-threat-analysis
python -m venv venv
venv\Scripts\Activate.ps1       # Windows
# source venv/bin/activate      # Linux/macOS
python -m pip install -r requirements.txt
```

### Quantum Inspire authentication (Experiment 2 only)

```bash
qi login
```

One-time browser-based authentication. Credentials are stored in `~/.quantuminspire/config.json`.

---

## Running the experiments

### Experiment 1 — Calibration Paradox

```bash
cd experiments
python experiment1_calibration_paradox.py
```

Requires MLPM hidden state cache (`.npz` files) pre-computed by `eval_mlpm_with_scores.py`. Results saved to `results/experiment1/`.

### Experiment 2 — Grover on Quantum Inspire

```bash
cd experiments

# Local Qiskit Aer simulator (no QI account needed):
python experiment2_grover_quantum_inspire.py --backend local

# Quantum Inspire QX Emulator:
python experiment2_grover_quantum_inspire.py --backend qi_sim
```

Results saved to `results/experiment2/`.

---

## Key results

### Experiment 1

Empirical confirmation that bypass density ρ(τ) is monotonically decreasing in τ across all four models, with quantum speedup factor S = √(1/ρ) growing as ρ decreases — consistent with the Calibration Paradox.

### Experiment 2

| k (iterations) | P_emp | P_theoretical | P_classical |
|---|---|---|---|
| 0 | 0.125 | 0.125 | 0.125 |
| 1 | 0.781 | 0.781 | 0.250 |
| **2** | **0.940** | **0.945** | 0.375 |
| 3 | 0.796 | 0.796 | 0.500 |

Optimal k*=2 (theoretical: π/4·√8 ≈ 2.22, floor=2). Error at k*: 0.53%.

---

## Dependencies

See `requirements.txt`. Key packages:

- `transformers`, `torch` — SLM inference
- `datasets` — WildGuardMix calibration data (`allenai/wildguardmix`)
- `qiskit`, `qiskit-aer`, `qiskit-quantuminspire` — quantum circuit simulation
- `numpy`, `pandas`, `scipy` — numerical analysis

---

## License

MIT License — see `LICENSE`.
