# Tensor-Guided Masked Diffusion for Logical Sequence Reasoning

A framework for combining **masked diffusion-style denoising**, **logical tensor-network (TN) reasoning**, and **TT-based adaptive mask policies** to repair partially masked symbolic sequences.

---

## Core Idea

Given a partially masked symbolic sequence

```
x^{(k)} ∈ (A ∪ {MASK})^n
```

with finite domain A, we iteratively repair it by combining:

1. **Denoiser** — produces a proposal distribution over masked positions
2. **Tensor-network marginals** — compute exact logical posteriors over masked positions given observed ones
3. **Mask policy** — selects which positions to unmask (or remask) at each step

The central mechanism is a **product-of-experts** correction:

```
p̃_i(v) ∝ p_i(v)^β · q_i(v)^{1-β}
```

where `p_i(v)` is the denoiser's proposal and `q_i(v)` is the TN marginal for variable `i` taking value `v`.

---

## Project Structure

```
tensor_diffusion_reasoning/
├── README.md
├── pyproject.toml
├── configs/              # YAML configs for domains and experiments
├── src/tdr/
│   ├── __init__.py       # Package root with MASK sentinel
│   ├── domains/          # Finite reasoning domain definitions
│   │   ├── base.py       # FiniteReasoningDomain, Factor, VerifierDiagnostics
│   │   └── sudoku4.py    # 4×4 Sudoku domain (288 solutions)
│   ├── tn/               # Tensor-network backends
│   │   └── brute_force_backend.py  # Exact brute-force marginal oracle
│   ├── diffusion/        # Masked diffusion loop
│   │   ├── state.py      # DiffusionState dataclass
│   │   ├── denoisers.py  # Random, local heuristic, TN-marginal denoisers
│   │   └── sampler.py    # MaskedDiffusionSampler core loop
│   ├── policies/         # Mask selection policies
│   │   └── entropy_policy.py  # Confidence, random, all-at-once policies
│   ├── verifier/         # Constraint verification
│   │   └── local_residuals.py
│   ├── experiments/      # Experiment runners
│   │   └── run_symbolic_tn_denoising.py
│   ├── training/         # Training loops (future)
│   └── utils/            # Logging, plotting, serialization
├── tests/
│   ├── test_sudoku4.py
│   └── test_marginals.py
└── results/              # Experiment outputs (logs, plots, checkpoints)
```

---

## Mathematical Formulation

### Masked State

A **masked state** is a vector

```
x_i ∈ {-1} ∪ {0, 1, ..., d-1}
```

where `MASK = -1` indicates an unobserved position and `0, ..., d-1` are valid domain values.

### Constraint Satisfaction

The system is governed by a set of factor constraints:

```
Ψ(x) = ∏_{G ∈ G} ψ_G(x_G)
```

Each factor `ψ_G(x_G)` is an indicator over group `G`; it is 1 if the group satisfies its constraint and 0 otherwise.
For 4×4 Sudoku there are 12 groups (4 rows + 4 columns + 4 boxes), each requiring all values distinct.

### Global Violation

```
V(x) = |{G ∈ G : ψ_G(x_G) = 0}|
```

counts the number of violated constraint groups.

### Local Residual

```
r_i(x) = |{G ∈ G : i ∈ G and ψ_G(x_G) = 0}|
```

is the number of violated groups that contain variable `i`. Used for verifier-guided remasking.

### Conditional Marginals

Given observed positions `Ω`, the conditional marginal over masked variable `i` is:

```
q_i(v) = P_Ψ(x_i = v | x_Ω) = Σ_{x_{M\{i}}} Ψ(x_Ω, x_i=v, x_{M\{i}}) / Σ_{x_M} Ψ(x_Ω, x_M)
```

These are computed exactly via brute-force enumeration for small domains (288 precomputed 4×4 Sudoku solutions).

### Denoisers

| Denoiser | Definition |
|----------|-----------|
| Random | q_i(v) = 1/d (uniform) for masked i |
| Local heuristic | Uniform over values not locally forbidden by observed neighbors |
| TN marginal | Exact q_i(v) from brute-force or tensor-network backend |

### Product-of-Experts Correction

```
log p̃_i(v) = β · log p_i(v) + (1-β) · log q_i(v) - log Z_i
```

Sweeping β ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0} interpolates between pure logic (β=0) and pure noisy denoising (β=1).

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Brute-force Sudoku 4×4 backend | ✅ Done |
| 2 | Tensor-factor backend (tnreason) | ⬜ |
| 3 | Product-of-experts denoising | ⬜ |
| 4 | Verifier-guided repair | ⬜ |
| 5 | tinyTT mask policy | ⬜ |
| 6 | Learned masked denoiser | ⬜ |
| 7 | Larger tasks (SAT, typed expressions) | ⬜ |

See [`tensor_guided_masked_diffusion_implementation_plan.md`](tensor_guided_masked_diffusion_implementation_plan.md)
for the full roadmap.

---

## Quick Start

```bash
# Install
cd tensor_diffusion_reasoning
uv pip install -e .

# Run tests
source ~/work/venv/python-ml/bin/activate
python -m pytest tests/ -v

# Run experiment
python -m tdr.experiments.run_symbolic_tn_denoising --trials 100 --mask-ratio 0.5
```

### Example Results (50% masked, 100 trials)

| Method | Success | Avg Steps | Final Violation |
|--------|---------|-----------|-----------------|
| Random | 0% | 20.0 | 10.4 |
| Local heuristic | 100% | 2.6 | 0.0 |
| TN marginal | 100% | 1.4 | 0.0 |

---

## Key Design Decisions

1. **Integer-coded variables** — values are 0, 1, ..., d-1 internally; MASK = -1
2. **Brute-force oracle first** — exact marginals for debugging before approximate TN methods
3. **Soft partial verifier** — a group is violated if two observed variables share a value (Option B)
4. **Candidate-set mask training** — regression over candidate masks, not policy gradient
5. **Avoid early LLM integration** — start with synthetic structured tasks

---

## References

- Implementation plan: [`tensor_guided_masked_diffusion_implementation_plan.md`](tensor_guided_masked_diffusion_implementation_plan.md)
- CTT (arXiv:2512.18059)
- tnreason: logical tensor-network inference
- tinyTT: low-rank tensor-train operations
