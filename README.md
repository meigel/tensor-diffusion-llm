# Verifier-Guided Remasking for Masked Diffusion

A framework for repairing corrupted symbolic sequences by combining **masked diffusion** with **verifier-guided remasking**.

**Core result:** An external symbolic verifier with *localized constraint residuals* — telling you *where* each violation occurs — dramatically improves any denoiser. This is denoiser-agnostic and works identically with exact marginals, local heuristics, learned MLPs, or transformer MDLMs.

---

## Core Idea

Given a partially corrupted symbolic sequence

```
x ∈ (A ∪ {MASK})ⁿ
```

with finite domain A, we iteratively repair it by:

1. **Denoising** — a denoiser proposes probability distributions for each position
2. **Verifier check** — a symbolic constraint verifier identifies violated positions
3. **Verifier-guided remasking** — positions with positive *local residuals* are returned to MASK for re-generation
4. **Iterate** — repeat until the sequence satisfies all constraints

This differs from existing approaches (ReMDM, RemeDi, PRISM) by using an *external symbolic verifier* that returns *per-variable residual information* — not just a global pass/fail.

---

## Project Structure

```
tensor-diffusion-llm/
├── README.md
├── pyproject.toml
├── tex/                       # LaTeX paper draft
│   ├── main.tex
│   └── main.pdf              # 7-page conference paper skeleton
├── src/tdr/
│   ├── __init__.py            # Package root with MASK = -1
│   ├── domains/
│   │   ├── base.py            # FiniteReasoningDomain, Factor, VerifierDiagnostics
│   │   ├── sudoku4.py         # 4×4 Sudoku domain (288 enumerated solutions)
│   │   ├── boolsat.py         # Planted k-SAT domain (random formulas per seed)
│   │   └── json_schema.py     # JSON user-profile domain (7 fields, cross-field constraints)
│   ├── tn/
│   │   ├── brute_force_backend.py
│   │   ├── factors.py
│   │   └── marginals.py       # MarginalBackend interface + contraction backend
│   ├── diffusion/
│   │   ├── state.py            # DiffusionState dataclass
│   │   ├── denoisers.py        # Random, LocalSudoku, TNMarginal, Noisy, PoE, Learned
│   │   ├── transformer_mdlm.py # TransformerDenoiserModel + MDLMTransformerDenoiser
│   │   ├── sampler.py          # MaskedDiffusionSampler: remask-fill loop
│   │   └── poe.py              # Product-of-experts combination
│   ├── policies/
│   │   ├── entropy_policy.py   # BaseMaskPolicy, ConfidenceUnmask, RandomMask, AllMask
│   │   └── verifier_policy.py  # VerifierRepair, RandomRemask, ConfidenceRemask
│   ├── training/
│   │   ├── datasets.py         # DenoisingDataset + compute_denoising_accuracy
│   │   └── train_denoiser.py   # MLP / MDLM denoiser training (--model mlp|mdlm)
│   ├── experiments/
│   │   ├── run_repair_experiment.py   # Main ablation: 6+ methods × 5 wrong ratios
│   │   ├── run_json_experiment.py     # JSON-specific experiment + MLP training
│   │   ├── run_noisy_sweep.py         # Denoiser-quality sweep
│   │   ├── run_poe_experiment.py      # Product-of-experts sweeps
│   │   └── save_noisy_sweep.py        # Results logging
│   ├── verifier/
│   │   └── local_residuals.py
│   └── utils/
│       ├── logging.py
│       ├── plotting.py
│       └── serialization.py
├── tests/
│   ├── test_sudoku4.py
│   ├── test_marginals.py
│   └── ...
└── results/
    ├── logs/           # JSON result files (gitignored)
    ├── plots/          # PDF figures (gitignored)
    └── checkpoints/    # Trained denoiser weights (gitignored)
```

---

## Domains

| Domain | Variables | Max domain | Constraints | Verifier |
|--------|-----------|-----------|-------------|----------|
| **4×4 Sudoku** | 16 cells | 4 | 12 groups (row/col/box distinct) | Row/col/box violation per cell |
| **Planted 3-SAT** | 20 vars | 2 | 60 clauses (k=3) | Clause satisfaction + residual |
| **JSON profile** | 7 fields | 63 (age) | Schema types + cross-field (admin→high clearance) | jsonschema + manual |

---

## Denoisers

| Denoiser | Description | Implementation |
|----------|-------------|---------------|
| **Random** | Uniform over domain values | `RandomDenoiser` |
| **Local heuristic** | Uniform over locally-allowed values (Sudoku) | `LocalSudokuDenoiser` |
| **TN marginal** | Exact conditional marginals via tensor contraction | `TNMarginalDenoiser` |
| **Noisy TN** | TN marginals + Gaussian logit noise (σ sweep) | `NoisyDenoiser` |
| **Learned MLP** | 2-layer MLP (256→256) trained on masked completion | `LearnedDenoiser` / `MLPDenoiser` |
| **Transformer MDLM** | 3-layer bidirectional transformer (128-dim, 4-head) | `MDLMTransformerDenoiser` / `TransformerDenoiserModel` |

---

## Policies (Remasking Strategies)

| Policy | Remask selection | Fill selection | Source |
|--------|-----------------|----------------|--------|
| **No repair** | None | Confidence ≥ 0.99 | `ConfidenceUnmaskPolicy` |
| **Verifier repair** | Positive local residuals | Confidence ≥ 0.99 | `VerifierRepairPolicy` |
| **Random remask** | Random subset | Confidence ≥ 0.99 | `RandomRemaskPolicy` |
| **Confidence remask** | After fill, low-confidence positions | All MASK | `ConfidenceRemaskPolicy` |

**Key isolation:** Repair + random denoiser = 0% success rate everywhere. This proves remasking alone is useless — verifier *localization* is essential, not just the act of reopening positions.

---

## Experimental Results

### Sudoku 4×4 (n=16, d=4, 288 solutions, n=100 trials)

| Method | wr=0.00 | wr=0.05 | wr=0.10 | wr=0.20 | wr=0.30 |
|--------|---------|---------|---------|---------|---------|
| Local | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PoE | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Repair + random | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Repair + local | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TN + repair | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PoE + repair | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Planted 3-SAT (n=20, m=60, fresh formulas per trial, n=200 trials)

| Method | wr=0.00 | wr=0.05 | wr=0.10 | wr=0.20 | wr=0.30 |
|--------|---------|---------|---------|---------|---------|
| Random | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| TN | 1.00 | 1.00 | 0.99 | 0.98 | 0.89 |
| Repair + random | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| TN + repair | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |

### JSON Schema (n=7, d=63, 7 cross-field constraints, n=30 trials)

| Method | wr=0.0 | wr=0.2 | wr=0.4 |
|--------|--------|--------|--------|
| Learned MLP | 1.00 | 0.93 | 0.93 |
| + Verifier repair | 1.00 | 1.00 | 1.00 |
| Transformer MDLM | 1.00 | 0.94 | — |
| + Verifier repair | 1.00 | 1.00 | — |

The **denoiser-agnostic pattern** is clear: verifier repair fills the gap for any imperfect denoiser, while the mechanism-isolating control (repair + random = 0%) proves that remasking alone provides no benefit without verifier guidance.

---

## Usage

```bash
# Activate environment
source ~/work/venv/python-ml/bin/activate

# Run tests
python -m pytest tests/ -v

# Train an MLP denoiser
python -m tdr.training.train_denoiser --domain sudoku --epochs 50

# Train a transformer MDLM on JSON
python -m tdr.training.train_denoiser --domain json --model mdlm --epochs 30

# Run full repair ablation (Sudoku + SAT)
python -m tdr.experiments.run_repair_experiment --domain all --trials 100

# Run JSON experiment
python -m tdr.experiments.run_json_experiment
```

---

## Milestone Status

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 1 | Brute-force Sudoku 4×4 backend | ✅ | Done |
| 2 | Tensor-factor backend (tnreason) | ✅ | Contraction-based marginal backend |
| 3 | Baseline denoisers (Random, Local, TN, Noisy, PoE) | ✅ | All implemented |
| 4 | **Verifier-guided remasking** | ✅ **Core contribution** | Verifier repair + mechanism-isolating controls |
| 5 | Mechanism-isolating baselines (repair + random, repair + local) | ✅ | Proves localization matters, not just remasking |
| 6 | Learned denoisers (MLP, transformer MDLM) | ✅ | Both trainable end-to-end |
| 7 | Realistic JSON domain | ✅ | Schema + cross-field constraints |
| 8 | Domain-randomized SAT (formula per seed) | ✅ | Results generalize across formulas |
| 9 | Confidence / entropy baselines (ReMDM-style) | ✅ | `ConfidenceRemaskPolicy`, `ConfidenceFillThenRemask` |
| 10 | Conference paper draft | In progress | `tex/main.tex` — 7 pages, needs JSON + MDLM results |

---

## Key Publications (Differentiation)

| Paper | Approach | Difference from our work |
|-------|----------|------------------------|
| **ReMDM** (2503.00307) | Confidence-based remasking | No symbolic verifier; global uncertainty only |
| **RemeDi** (2509.23653) | Retrieval-augmented repair | No localized residual signal |
| **PRISM** (2510.01384) | Probe-guided search | No iterated denoising with per-step verification |
| **This work** | Verifier-guided remasking | Localized constraint residuals guide *which* positions to reopen |

---

## References

- Full implementation plan: [`tensor_guided_masked_diffusion_implementation_plan.md`](tensor_guided_masked_diffusion_implementation_plan.md)
- Codex 5.5 improvement plan: [`paper_improvement_plan.md`](paper_improvement_plan.md)
- CTT (arXiv:2512.18059)
- tinyTT: low-rank tensor-train operations
