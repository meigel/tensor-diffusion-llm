# Implementation Plan: Tensor-Guided Masked Diffusion for Logical Sequence Reasoning

This document gives a staged implementation roadmap for combining **masked diffusion-style denoising**, **logical tensor-network reasoning**, and **TT-based adaptive mask policies**, starting with `tnreason` and `tinyTT`.

The plan is structured so that useful results can be obtained before integrating a real diffusion LLM.

---

## 0. Core Goal

Build a small experimental framework where a partially masked symbolic sequence

\[
x^{(k)} \in (\mathcal A \cup \{\mathtt{MASK}\})^n
\]

is iteratively repaired by combining:

\[
\text{neural/noisy denoiser}
+
\text{logical tensor-network marginals}
+
\text{adaptive remasking policy}.
\]

The first prototype should answer:

1. Can tensor-network marginals act as a diffusion-style denoiser?
2. Do TN marginals improve a noisy denoiser?
3. Can a TT mask policy learn nonlocal remasking decisions?
4. Does verifier-guided remasking reduce constraint violation faster than local entropy/confidence heuristics?

---

## 1. Recommended Project Structure

Use a small Python package layout:

```text
tensor_diffusion_reasoning/
│
├── README.md
├── pyproject.toml
├── configs/
│   ├── sudoku4.yaml
│   ├── boolsat.yaml
│   ├── typed_expr.yaml
│   └── mask_policy.yaml
│
├── src/
│   └── tdr/
│       ├── __init__.py
│       │
│       ├── domains/
│       │   ├── base.py
│       │   ├── sudoku4.py
│       │   ├── boolsat.py
│       │   ├── typed_expr.py
│       │   └── json_repair.py
│       │
│       ├── tn/
│       │   ├── tnreason_backend.py
│       │   ├── brute_force_backend.py
│       │   ├── marginals.py
│       │   └── factors.py
│       │
│       ├── diffusion/
│       │   ├── state.py
│       │   ├── corruption.py
│       │   ├── denoisers.py
│       │   ├── schedules.py
│       │   └── sampler.py
│       │
│       ├── policies/
│       │   ├── base.py
│       │   ├── random_policy.py
│       │   ├── entropy_policy.py
│       │   ├── verifier_policy.py
│       │   ├── tn_marginal_policy.py
│       │   └── tinytt_policy.py
│       │
│       ├── verifier/
│       │   ├── base.py
│       │   ├── local_residuals.py
│       │   └── metrics.py
│       │
│       ├── training/
│       │   ├── train_denoiser.py
│       │   ├── train_mask_policy.py
│       │   └── datasets.py
│       │
│       ├── experiments/
│       │   ├── run_symbolic_tn_denoising.py
│       │   ├── run_noisy_denoiser_poe.py
│       │   ├── run_tt_mask_policy.py
│       │   └── run_ablation.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── plotting.py
│           ├── seeds.py
│           └── serialization.py
│
├── tests/
│   ├── test_sudoku4.py
│   ├── test_boolsat.py
│   ├── test_marginals.py
│   ├── test_sampler.py
│   ├── test_policies.py
│   └── test_tinytt_policy.py
│
└── results/
    ├── logs/
    ├── plots/
    └── checkpoints/
```

Start with a **brute-force backend** for small domains even if the final goal is `tnreason`. It gives an oracle for debugging the tensor-network backend.

---

## 2. Main Abstraction: Finite-Domain Masked Reasoning Problem

Define a base class representing a discrete logical completion problem.

```python
class FiniteReasoningDomain:
    """
    Base class for finite symbolic completion tasks.
    """

    def num_variables(self) -> int:
        raise NotImplementedError

    def domain_size(self, i: int) -> int:
        raise NotImplementedError

    def sample_solution(self, rng):
        """
        Return a valid full assignment x of shape (n,).
        """
        raise NotImplementedError

    def corrupt(self, x, mask_ratio: float, rng):
        """
        Return masked assignment x_masked and mask vector.
        """
        raise NotImplementedError

    def verifier(self, x):
        """
        Return global violation and local residual information.
        """
        raise NotImplementedError

    def build_factors(self):
        """
        Return logical factors for tensor-network or brute-force inference.
        """
        raise NotImplementedError
```

The state should use integer-coded variables.

```python
MASK = -1

# x[i] = 0,1,2,3 for Sudoku 4x4
# x[i] = -1 means masked
```

A masked state is:

\[
x_i =
\begin{cases}
a_i \in \mathcal A_i, & \text{observed},\\
-1, & \text{masked}.
\end{cases}
\]

---

## 3. First Domain: 4x4 Sudoku

Start with \(4\times4\) Sudoku, not \(9\times9\). It is small enough for brute force and structured enough to require nonlocal constraints.

### 3.1 Variables

There are

\[
n = 16
\]

variables:

\[
x_{r,c} \in \{1,2,3,4\}.
\]

Use zero-based values internally:

\[
x_{r,c} \in \{0,1,2,3\}.
\]

Flatten index:

\[
i = 4r+c.
\]

### 3.2 Constraints

Rows, columns, and \(2\times2\) boxes must contain all values exactly once.

For a group \(G\subset\{1,\dots,16\}\),

\[
\psi_G(x_G)
=
\mathbf 1\{x_i : i\in G\} = \{0,1,2,3\}.
\]

The full constraint score is:

\[
\Psi(x)
=
\prod_{G\in\mathcal G}
\psi_G(x_G).
\]

For \(4\times4\), there are:

- 4 row factors,
- 4 column factors,
- 4 box factors.

Each factor has arity 4 and shape \(4^4\).

### 3.3 Verifier

The verifier should return both global and local residuals.

Global violation:

\[
V(x)
=
\sum_{G\in\mathcal G}
\mathbf 1\{\text{group }G\text{ violates Sudoku constraint}\}.
\]

For masked entries, use either:

**Option A: ignore masked variables.**

Only evaluate groups with no masks.

**Option B: soft partial violation.**

A group is violated if two observed variables already have the same value.

For diffusion-style repair, use Option B.

Local residual:

\[
r_i(x)
=
\#\{\text{violated groups containing }i\}.
\]

This gives a remasking signal.

---

## 4. Tensor-Network Marginal Backend

At each masked state \(x_\Omega\), compute logical marginals over masked variables:

\[
q_i(v)
=
\mathbb P_{\Psi}(x_i=v\mid x_\Omega),
\qquad i\notin\Omega.
\]

For Sudoku:

\[
q_i(v)
=
\frac{
\sum_{x_{M\setminus\{i\}}}
\Psi(x_\Omega, x_i=v, x_{M\setminus\{i\}})
}{
\sum_{x_M}
\Psi(x_\Omega, x_M)
}.
\]

### 4.1 Brute-Force Backend

For \(4\times4\), start with brute force over all solutions.

Precompute all valid Sudoku solutions:

```python
solutions = np.array([...])  # shape (num_solutions, 16)
```

Then conditioning is trivial:

```python
def brute_force_marginals(solutions, x_masked):
    compatible = np.ones(len(solutions), dtype=bool)
    for i, xi in enumerate(x_masked):
        if xi != MASK:
            compatible &= (solutions[:, i] == xi)

    sol = solutions[compatible]
    if len(sol) == 0:
        return None, 0

    q = np.zeros((16, 4))
    for i in range(16):
        for v in range(4):
            q[i, v] = np.mean(sol[:, i] == v)
    return q, len(sol)
```

This gives an exact oracle.

### 4.2 tnreason Backend

Once brute force works, implement a backend that builds tensor factors and contracts them.

Interface:

```python
class MarginalBackend:
    def marginals(self, domain, x_masked):
        """
        Returns:
            q: array shape (n, max_domain_size)
            logZ or Z
            status: ok / contradiction / approximate
        """
```

Implementation target:

```python
class TNReasonBackend(MarginalBackend):
    def __init__(self, contraction_method="auto"):
        ...

    def marginals(self, domain, x_masked):
        factors = domain.build_factors()
        conditioned_factors = condition_factors(factors, x_masked)
        q = contract_for_all_single_site_marginals(conditioned_factors)
        return q, logZ, status
```

The precise `tnreason` API may require adaptation, but the internal interface should remain stable.

### 4.3 Debug Test

For random partial Sudoku states:

\[
\|q^{\mathrm{brute}} - q^{\mathrm{tn}}\|_\infty < 10^{-10}
\]

for exact contraction.

Test file:

```text
tests/test_marginals.py
```

---

## 5. Diffusion-Style Symbolic Sampler

Implement a symbolic masked diffusion loop independent of neural networks.

### 5.1 State

```python
@dataclass
class DiffusionState:
    x: np.ndarray          # shape (n,), entries in {0,...,d-1} or MASK
    step: int
    history: list
```

### 5.2 General Sampler

```python
class MaskedDiffusionSampler:
    def __init__(self, denoiser, mask_policy, verifier, max_steps):
        self.denoiser = denoiser
        self.mask_policy = mask_policy
        self.verifier = verifier
        self.max_steps = max_steps

    def run(self, x_init, rng):
        state = DiffusionState(x=x_init.copy(), step=0, history=[])

        for k in range(self.max_steps):
            diagnostics = self.verifier(state.x)

            proposal_dist = self.denoiser.predict(state.x)
            x_filled = fill_some_positions(
                state.x,
                proposal_dist,
                rng
            )

            diagnostics_new = self.verifier(x_filled)

            remask = self.mask_policy.select_mask(
                x_filled,
                proposal_dist,
                diagnostics_new
            )

            x_next = x_filled.copy()
            x_next[remask] = MASK

            state.history.append({
                "x": state.x.copy(),
                "proposal": x_filled.copy(),
                "violation": diagnostics_new.global_violation,
                "num_masks": np.sum(x_next == MASK),
            })

            state.x = x_next
            state.step += 1

            if is_complete(state.x) and diagnostics_new.global_violation == 0:
                break

        return state
```

There are two modes:

### Mode 1: Monotone Unmasking

Once a token is filled, it stays fixed unless the verifier says it is wrong.

### Mode 2: Full Remasking

The policy may remask any position, including previously filled ones.

For early experiments, use Mode 1. For repair experiments, use Mode 2.

---

## 6. Denoisers

Implement denoisers in increasing complexity.

### 6.1 TN Marginal Denoiser

The pure tensor-network denoiser uses:

\[
p_i(v) = q_i(v).
\]

```python
class TNMarginalDenoiser:
    def __init__(self, marginal_backend):
        self.backend = marginal_backend

    def predict(self, x_masked):
        q, logZ, status = self.backend.marginals(x_masked)
        return q
```

Filling rule:

\[
\hat x_i = \arg\max_v q_i(v)
\]

for positions with

\[
\max_v q_i(v) \ge \tau.
\]

### 6.2 Noisy Denoiser

This emulates an imperfect neural model.

Given exact marginal \(q_i\), define:

\[
p_i(v)
=
(1-\epsilon)q_i(v)
+
\epsilon u_i(v),
\]

where \(u_i\) is uniform.

Or use logit noise:

\[
\ell_i(v) = \log(q_i(v)+\delta) + \sigma \xi_{i,v},
\]

\[
p_i(v) = \operatorname{softmax}(\ell_i)_v.
\]

This lets you test robustness before training a neural model.

### 6.3 Local Heuristic Denoiser

For Sudoku, use only row/column/box local allowed values.

This gives a weak baseline:

\[
p_i(v)
\propto
\mathbf 1\{v\text{ does not immediately violate observed neighbors}\}.
\]

It does not reason globally.

### 6.4 Learned Denoiser

Train a small MLP or transformer on masked solutions.

Input:

\[
x_{\mathrm{masked}} \in \{-1,0,1,2,3\}^{16}.
\]

Represent each variable by embedding:

- value embedding for 0,1,2,3;
- mask embedding for -1;
- position embedding.

Model options:

#### Small MLP

Flatten embeddings and predict logits for all positions.

#### Tiny Transformer

Bidirectional transformer over 16 tokens.

Loss:

\[
\mathcal L_{\mathrm{denoise}}
=
-\sum_{i\in M}
\log p_\theta(x_i\mid x_{\mathrm{masked}}).
\]

This is already a masked diffusion LM analogue.

---

## 7. Product-of-Experts Correction

This is the central mechanism for combining neural/noisy denoising with tensor-network logic.

Given neural distribution \(p_i(v)\) and TN marginal \(q_i(v)\):

\[
\tilde p_i(v)
\propto
p_i(v)^\beta q_i(v)^{1-\beta}.
\]

Equivalent log form:

\[
\log \tilde p_i(v)
=
\beta\log p_i(v)
+
(1-\beta)\log q_i(v)
-
\log Z_i.
\]

Implementation:

```python
def product_of_experts(p, q, beta=0.5, eps=1e-12):
    logp = np.log(p + eps)
    logq = np.log(q + eps)
    logits = beta * logp + (1.0 - beta) * logq
    logits -= logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    out /= out.sum(axis=-1, keepdims=True)
    return out
```

Experiments sweep:

\[
\beta \in \{0,0.1,0.25,0.5,0.75,0.9,1\}.
\]

Interpretation:

- \(\beta=0\): pure logic/TN;
- \(\beta=1\): pure neural/noisy denoiser;
- intermediate: logic-guided diffusion.

---

## 8. Mask Policies

The mask policy determines which positions remain masked or are remasked.

### 8.1 Random Policy

Baseline.

```python
class RandomMaskPolicy:
    def select_mask(self, x, dist, diagnostics):
        ...
```

### 8.2 Confidence Policy

Fill high-confidence variables.

Let

\[
c_i = \max_v p_i(v).
\]

Unmask if:

\[
c_i \ge \tau.
\]

Remask if:

\[
c_i < \tau_{\mathrm{low}}.
\]

### 8.3 Entropy Policy

Entropy:

\[
H_i = -\sum_v p_i(v)\log p_i(v).
\]

Unmask lowest-entropy positions.

### 8.4 Verifier-Local Policy

Use local residuals \(r_i(x)\).

Remask positions with

\[
r_i(x) > 0.
\]

For Sudoku:

\[
r_i = \#\{\text{violated row/column/box constraints involving }i\}.
\]

This is a strong baseline.

### 8.5 TN Marginal Policy

Use TN marginal sharpness:

\[
s_i = \max_v q_i(v).
\]

Fill positions with high \(s_i\).

Contradiction detection:

If

\[
Z(x_\Omega)=0,
\]

the current observed partial assignment is inconsistent. Then remask positions with high local residual, or use a minimal-conflict heuristic.

### 8.6 tinyTT Policy

This is the more research-oriented part.

The policy scores binary masks:

\[
m\in\{0,1\}^n,
\]

where \(m_i=1\) means “mask or repair position \(i\).”

Score:

\[
S_\eta(m;f)
=
S_{\mathrm{local}}(m;f)
+
S_{\mathrm{TT}}(m).
\]

Local part:

\[
S_{\mathrm{local}}(m;f)
=
\sum_i a_\eta(f_i,m_i).
\]

TT part:

\[
S_{\mathrm{TT}}(m)
=
G_1(m_1)G_2(m_2)\cdots G_n(m_n),
\]

where the scalar is obtained by contracting TT cores.

A simple first implementation ignores \(f\) inside the TT and uses \(f\) only in local logits. Later, use feature-conditioned TT cores.

---

## 9. tinyTT Mask Policy Design

### 9.1 Candidate-Set Approach

Do not normalize over all \(2^n\) masks initially.

For each state \(x\), generate a candidate set:

\[
\mathcal C(x)
=
\{m^{(1)},\ldots,m^{(B)}\}.
\]

Include:

1. oracle mask,
2. random masks,
3. entropy masks,
4. verifier masks,
5. local perturbations of oracle mask.

Train by softmax over candidate scores:

\[
\mathcal L(\eta)
=
-\log
\frac{\exp S_\eta(m_\star;x)}
{\sum_{m\in\mathcal C(x)} \exp S_\eta(m;x)}.
\]

This avoids full partition functions.

### 9.2 Oracle Mask Generation

For a corrupted state \(x\), define oracle repair set:

\[
M_\star(x)
=
\{i : x_i \ne x_i^{\mathrm{true}}\}.
\]

This is available in synthetic data.

For verifier-only oracle:

\[
M_\star(x)
=
\{i : r_i(x)>0\}.
\]

The true-difference oracle is stronger but less realistic. Use both.

### 9.3 TT Score Implementation

Assume binary masks \(m_i\in\{0,1\}\).

TT cores:

\[
G_i \in \mathbb R^{r_{i-1}\times 2 \times r_i}.
\]

Score:

\[
S_{\mathrm{TT}}(m)
=
G_1[:,m_1,:]
G_2[:,m_2,:]
\cdots
G_n[:,m_n,:].
\]

In code:

```python
def tt_score(cores, m):
    vec = cores[0][0, m[0], :]  # shape (r1,)
    for i in range(1, len(m)):
        vec = vec @ cores[i][:, m[i], :]
    return vec.item()
```

Use `tinyTT` if it already has TT contraction utilities; otherwise implement this small scoring routine directly and later replace it with `tinyTT`.

### 9.4 Feature-Conditioned Extension

Let features \(f_i\in\mathbb R^d\).

Define local feature map:

\[
\phi_i(f_i,m_i)
\in \mathbb R^{p}.
\]

Core:

\[
G_i(f_i,m_i)
=
\sum_{\ell=1}^p
\phi_{i,\ell}(f_i,m_i)
A_{i,\ell}.
\]

Then

\[
G_i(f_i,m_i)
\in\mathbb R^{r_{i-1}\times r_i}.
\]

This is a tensor-train recurrent score model over the mask.

Do this only after the candidate-set static TT works.

---

## 10. Training Datasets

Generate synthetic datasets of partial states.

Each sample:

```python
{
    "x_true": full valid assignment,
    "x_corrupt": corrupted assignment,
    "x_masked": masked assignment,
    "mask_initial": initial mask,
    "oracle_repair_mask": binary vector,
    "verifier_residual": local residual vector,
    "features": per-position features,
}
```

### 10.1 Corruption Types

Use several corruption modes.

#### Mask-Only Corruption

Start from valid solution, mask positions.

\[
x_i =
\begin{cases}
x_i^{\mathrm{true}}, & i\notin M,\\
\mathtt{MASK}, & i\in M.
\end{cases}
\]

#### Wrong-Token Corruption

Replace some variables by wrong values.

\[
x_i \sim \operatorname{Uniform}(\mathcal A_i\setminus\{x_i^{\mathrm{true}}\}).
\]

#### Mixed Corruption

Mask some variables and corrupt others.

This is important for remasking / repair.

### 10.2 Difficulty Levels

For Sudoku 4x4:

- easy: 25% masked, no wrong tokens;
- medium: 50% masked;
- hard: 75% masked;
- repair: 50% masked + 10–20% wrong tokens;
- contradiction-heavy: wrong tokens chosen to violate multiple constraints.

---

## 11. Experiment Suite

### Experiment 1: Pure TN Denoising

#### Purpose

Show that logical tensor-network marginals can act as a masked denoiser.

#### Methods

Compare:

1. random filling,
2. local allowed-values heuristic,
3. brute-force exact marginals,
4. `tnreason` marginals.

#### Metrics

- final success rate,
- average number of steps,
- violation decay,
- marginal entropy decay,
- contradiction rate,
- cost per step.

#### Expected Result

TN marginals should solve small structured tasks efficiently when constraints identify the solution.

---

### Experiment 2: Noisy Denoiser + TN Product-of-Experts

#### Purpose

Test whether TN logic improves imperfect denoisers.

#### Methods

For different noise levels \(\sigma\), compare:

1. noisy denoiser only,
2. TN marginal only,
3. product-of-experts.

Sweep:

\[
\sigma \in \{0,0.25,0.5,1.0,2.0\},
\]

\[
\beta \in \{0,0.1,0.25,0.5,0.75,0.9,1.0\}.
\]

#### Metrics

- success rate,
- invalid completion rate,
- average violation,
- number of steps,
- sensitivity to \(\beta\).

#### Expected Result

Intermediate \(\beta\) should outperform pure noisy denoising when the TN backend is informative.

---

### Experiment 3: Learned Denoiser + TN Correction

#### Purpose

Replace synthetic noise with a learned masked model.

#### Model

Small bidirectional transformer:

- sequence length: 16,
- vocab size: 5 including mask,
- embedding dim: 64,
- layers: 2,
- heads: 4,
- FF dim: 128.

Loss:

\[
-\sum_{i\in M}\log p_\theta(x_i\mid x_{\mathrm{masked}}).
\]

#### Compare

1. learned denoiser only,
2. learned denoiser + product-of-experts TN correction,
3. learned denoiser + verifier remasking,
4. learned denoiser + TN correction + verifier remasking.

#### Expected Result

TN correction should improve out-of-distribution masking ratios and repair cases.

---

### Experiment 4: TT Mask Policy

#### Purpose

Test whether nonlocal mask selection can be learned in low TT rank.

#### Setup

Use mixed corruption states with wrong tokens.

Candidate masks:

- oracle true-difference mask,
- verifier residual mask,
- entropy mask,
- random same-size masks,
- perturbed masks.

Train TT candidate scorer.

#### Compare

1. entropy policy,
2. verifier-local policy,
3. TT policy,
4. TT + verifier features.

#### Metrics

- repair success,
- number of denoising cycles,
- residual decrease per cycle,
- mask precision/recall against oracle repair set,
- rank vs performance.

#### Rank Sweep

\[
r \in \{1,2,4,8,16\}.
\]

Rank \(1\) is essentially independent structure. Higher ranks capture dependencies.

#### Expected Result

TT should help when repair masks are nonlocal and structured. If it does not, the task is too easy or too local.

---

### Experiment 5: Larger Symbolic Tasks

Once 4x4 Sudoku works, move to a task where brute force fails.

Candidates:

1. \(6\times6\) Sudoku-like Latin square,
2. Boolean SAT with 30–100 variables,
3. typed expression completion,
4. JSON repair,
5. simple Python AST repair.

This is where `tnreason` or approximate TN contraction becomes meaningful.

---

## 12. Evaluation Metrics

Use a common metrics interface.

```python
@dataclass
class RunMetrics:
    success: bool
    final_violation: float
    num_steps: int
    num_contradictions: int
    avg_entropy: float
    wall_time: float
    masks_per_step: list[int]
    violation_per_step: list[float]
    entropy_per_step: list[float]
```

### 12.1 Global Violation

\[
V(x)
=
\sum_{a\in\mathcal F}
\mathbf 1\{\psi_a(x_{\partial a})=0\}.
\]

For soft factors:

\[
V(x)
=
\sum_{a\in\mathcal F}
-\log(\psi_a(x_{\partial a})+\epsilon).
\]

### 12.2 Entropy

\[
H^{(k)}
=
\frac1{|M_k|}
\sum_{i\in M_k}
\left(
-\sum_v p_i^{(k)}(v)\log p_i^{(k)}(v)
\right).
\]

### 12.3 Marginal Sharpness

\[
S^{(k)}
=
\frac1{|M_k|}
\sum_{i\in M_k}
\max_v q_i^{(k)}(v).
\]

### 12.4 Residual Decrease

\[
\Delta V_k
=
V(x^{(k)}) - V(x^{(k+1)}).
\]

Plot:

\[
\E[V(x^{(k)})]
\]

over runs.

---

## 13. Minimal Algorithms

### 13.1 Pure TN Masked Completion

```python
def pure_tn_completion(x_masked, backend, threshold=0.99, max_steps=20):
    x = x_masked.copy()
    history = []

    for k in range(max_steps):
        q, logZ, status = backend.marginals(x)

        if status == "contradiction":
            break

        changed = False
        for i in range(len(x)):
            if x[i] == MASK:
                v = np.argmax(q[i])
                conf = q[i, v]
                if conf >= threshold:
                    x[i] = v
                    changed = True

        history.append({
            "step": k,
            "x": x.copy(),
            "num_masked": np.sum(x == MASK),
            "logZ": logZ,
        })

        if np.all(x != MASK):
            break

        if not changed:
            # Fill one most confident variable to make progress.
            masked = np.where(x == MASK)[0]
            i = max(masked, key=lambda j: np.max(q[j]))
            x[i] = np.argmax(q[i])
            changed = True

    return x, history
```

### 13.2 Product-of-Experts Denoising

```python
def poe_completion(x_masked, neural_denoiser, tn_backend, beta, threshold, max_steps):
    x = x_masked.copy()
    history = []

    for k in range(max_steps):
        p = neural_denoiser.predict(x)
        q, logZ, status = tn_backend.marginals(x)

        if status == "contradiction":
            # remasking policy needed here
            pass

        p_combined = product_of_experts(p, q, beta=beta)

        masked = np.where(x == MASK)[0]
        if len(masked) == 0:
            break

        confidences = p_combined[masked].max(axis=1)
        selected = masked[confidences >= threshold]

        if len(selected) == 0:
            selected = [masked[np.argmax(confidences)]]

        for i in selected:
            x[i] = np.argmax(p_combined[i])

        history.append(...)

    return x, history
```

### 13.3 Verifier-Guided Repair Loop

```python
def repair_loop(x_init, denoiser, policy, verifier, max_steps):
    x = x_init.copy()
    history = []

    for k in range(max_steps):
        diagnostics = verifier(x)

        if diagnostics.global_violation == 0 and np.all(x != MASK):
            break

        # Mask positions selected for repair.
        repair_mask = policy.select_mask(x, diagnostics)
        x[repair_mask] = MASK

        # Predict distributions and fill.
        p = denoiser.predict(x)
        x = fill_by_confidence(x, p)

        history.append({
            "step": k,
            "violation": diagnostics.global_violation,
            "repair_size": repair_mask.sum(),
            "num_masked": np.sum(x == MASK),
        })

    return x, history
```

---

## 14. Concrete `tnreason` Usage Strategy

Because `tnreason` may use its own abstractions, avoid locking the rest of the code to its API.

Create a translation layer:

```python
class TNReasonCompiler:
    def compile_domain(self, domain):
        """
        Convert domain factors into tnreason representation.
        """

    def condition(self, compiled_network, x_masked):
        """
        Apply observed assignments.
        """

    def marginal(self, conditioned_network, variable_index):
        """
        Return marginal over one variable.
        """
```

For each domain factor, define a generic factor object first:

```python
@dataclass
class Factor:
    variables: tuple[int, ...]
    table: np.ndarray
```

For Sudoku group factor:

```python
def all_different_factor(vars, d=4):
    table = np.zeros((d,) * len(vars))
    for assignment in itertools.product(range(d), repeat=len(vars)):
        if len(set(assignment)) == len(vars):
            table[assignment] = 1.0
    return Factor(variables=tuple(vars), table=table)
```

Then either:

- contract with brute force,
- contract with `einsum`,
- compile to `tnreason`.

This gives reliable debugging.

---

## 15. Concrete `tinyTT` Usage Strategy

Use `tinyTT` first for TT tensor experiments, not full integration.

### 15.1 Standalone TT Mask Scorer

Represent a score tensor:

\[
S(m_1,\dots,m_n)
\]

with binary physical dimension.

Start with manually initialized TT cores.

Train with PyTorch or NumPy + finite differences. Prefer PyTorch for training unless `tinyTT` already provides autograd-compatible operations.

If `tinyTT` is NumPy-only, then use:

1. alternating least squares,
2. small candidate-set regression,
3. or implement PyTorch TT scorer separately for the policy.

### 15.2 Candidate Regression Target

Given candidate masks \(m^{(b)}\), define target score:

\[
y_b = -V(\operatorname{Repair}(x,m^{(b)})).
\]

Fit:

\[
S_\eta(m^{(b)}) \approx y_b.
\]

Loss:

\[
\sum_b |S_\eta(m^{(b)})-y_b|^2.
\]

Then select:

\[
\hat m = \arg\max_{m\in\mathcal C(x)} S_\eta(m).
\]

This is simpler than policy-gradient training.

### 15.3 Rank Adaptivity

Start with rank \(r=1\). Increase if validation loss plateaus.

Ranks:

\[
1,2,4,8,16.
\]

Storage for binary physical dimension:

\[
\mathcal O(2nr^2).
\]

---

## 16. Initial Test Cases

### Test Case 1: Sudoku Completion

Input:

```text
. 2 . .
. . 3 .
. . . 1
4 . . .
```

Output should be a valid solution.

Use mask-only corruption first.

### Test Case 2: Sudoku Repair

Start from a valid Sudoku, corrupt two values, then remask detected conflict locations.

The difficulty is that a conflict usually implicates multiple positions. The policy must decide which one to remask.

This is where TT/nonlocal mask policies may matter.

### Test Case 3: Boolean SAT Completion

Variables:

\[
z_i\in\{0,1\}.
\]

Clauses:

\[
(z_1 \lor \neg z_3 \lor z_7)
\land
(\neg z_2 \lor z_4)
\land
\cdots
\]

Masked assignment completion:

- Some variables observed.
- Need complete satisfying assignment.

Tensor factors are clause indicators.

### Test Case 4: Typed Expression Completion

Grammar:

```text
expr ::= int
       | bool
       | expr + expr
       | expr == expr
       | if bool_expr then expr else expr
```

Constraints enforce type consistency.

This is closer to code and proof repair.

---

## 17. Logging and Plots

Use JSONL logs.

Each run logs:

```json
{
  "run_id": "...",
  "domain": "sudoku4",
  "method": "poe_beta_0.5_verifier_remask",
  "seed": 123,
  "mask_ratio": 0.5,
  "corruption_ratio": 0.1,
  "success": true,
  "num_steps": 7,
  "final_violation": 0,
  "wall_time": 0.012,
  "history": [
    {"step": 0, "violation": 5, "num_masked": 8, "entropy": 1.12},
    {"step": 1, "violation": 3, "num_masked": 6, "entropy": 0.91}
  ]
}
```

Required plots:

1. success rate vs mask ratio;
2. violation \(V(x^{(k)})\) vs step;
3. entropy vs step;
4. wall-clock vs problem size;
5. success rate vs \(\beta\);
6. TT rank vs policy performance;
7. contradiction rate vs corruption ratio.

---

## 18. Ablation Matrix

Run a compact ablation table.

| Method | Denoiser | TN marginals | Remasking | TT policy |
|---|---:|---:|---:|---:|
| Random | random | no | no | no |
| Local | local heuristic | no | no | no |
| TN-only | TN | yes | no | no |
| Noisy | noisy | no | no | no |
| PoE | noisy/neural | yes | no | no |
| PoE + verifier | noisy/neural | yes | verifier | no |
| PoE + TT | noisy/neural | yes | learned | yes |

For each method report:

\[
\text{success},\quad
\text{steps},\quad
\text{violation},\quad
\text{time}.
\]

---

## 19. Milestone Schedule

### Milestone 1: Brute-Force Sudoku Backend

Deliverables:

- `Sudoku4Domain`
- solution generator
- brute-force marginals
- verifier
- pure TN/brute-force completion loop
- tests against known puzzles

Success criterion:

\[
>95\%
\]

completion success for mask-only corruption at moderate mask ratios where the solution is uniquely determined.

---

### Milestone 2: Tensor Factor Backend

Deliverables:

- generic `Factor`
- `all_different_factor`
- factor conditioning
- exact contraction by enumeration or `einsum`
- `tnreason` compiler wrapper

Success criterion:

TN marginals match brute-force marginals:

\[
\|q^{\mathrm{TN}}-q^{\mathrm{brute}}\|_\infty < 10^{-8}.
\]

---

### Milestone 3: Product-of-Experts Denoising

Deliverables:

- noisy denoiser
- PoE combination
- beta sweeps
- plots

Success criterion:

PoE improves success rate over noisy denoiser for nonzero noise.

---

### Milestone 4: Verifier-Guided Repair

Deliverables:

- wrong-token corruption
- local residual verifier
- verifier remasking policy
- repair loop

Success criterion:

Verifier-guided remasking reduces final violation and contradiction rate compared with monotone unmasking.

---

### Milestone 5: tinyTT Mask Policy

Deliverables:

- candidate mask generator
- TT score function
- oracle mask dataset
- candidate-set training
- rank sweep

Success criterion:

TT policy outperforms local entropy policy on at least one task with nonlocal repair dependencies.

---

### Milestone 6: Learned Masked Denoiser

Deliverables:

- small transformer or MLP denoiser
- training dataset
- denoising loss
- evaluation with and without TN correction

Success criterion:

TN correction improves OOD masking or corruption performance.

---

### Milestone 7: Larger Task

Deliverables:

- Boolean SAT or typed expression domain
- approximate TN / `tnreason` inference
- same sampler pipeline

Success criterion:

The framework transfers beyond Sudoku.

---

## 20. Design Choices to Avoid Early

Avoid these initially:

1. **Full LLM integration**  
   Too much engineering noise.

2. **Tensorizing Transformer weights**  
   This is compression, not the conceptual contribution.

3. **Differentiable logic over BPE tokens**  
   Very brittle.

4. **9x9 Sudoku immediately**  
   Contraction and data generation become distracting.

5. **Policy-gradient training for masks**  
   Candidate-set ranking/regression is simpler and more stable.

6. **End-to-end training of everything**  
   First isolate the mechanism.

---

## 21. First Concrete Coding Task

Start with this minimal target:

```text
Implement 4x4 Sudoku masked completion using exact logical marginals.
```

Files:

```text
src/tdr/domains/sudoku4.py
src/tdr/tn/brute_force_backend.py
src/tdr/diffusion/sampler.py
src/tdr/policies/entropy_policy.py
src/tdr/verifier/local_residuals.py
src/tdr/experiments/run_symbolic_tn_denoising.py
tests/test_sudoku4.py
tests/test_marginals.py
```

The first experiment should output:

```text
method=random        success=...
method=local         success=...
method=tn_marginal   success=...
```

and produce a plot:

```text
results/plots/sudoku4_violation_decay.png
```

---

## 22. Minimal Pseudocode for the First Milestone

```python
def main():
    rng = np.random.default_rng(0)

    domain = Sudoku4Domain()
    backend = BruteForceMarginalBackend(domain)

    methods = {
        "random": RandomDenoiser(domain),
        "local": LocalSudokuDenoiser(domain),
        "tn": TNMarginalDenoiser(backend),
    }

    for method_name, denoiser in methods.items():
        successes = []
        violations = []

        for seed in range(100):
            rng = np.random.default_rng(seed)

            x_true = domain.sample_solution(rng)
            x_masked = domain.corrupt(x_true, mask_ratio=0.5, rng=rng)

            sampler = MaskedDiffusionSampler(
                denoiser=denoiser,
                mask_policy=ConfidenceUnmaskPolicy(threshold=0.99),
                verifier=domain.verifier,
                max_steps=20,
            )

            result = sampler.run(x_masked, rng)

            v = domain.verifier(result.x).global_violation
            success = (v == 0) and np.all(result.x != MASK)

            successes.append(success)
            violations.append(v)

        print(method_name, np.mean(successes), np.mean(violations))
```

---

## 23. What Would Count as an Interesting Early Result?

An early result is worth writing up if one of the following is observed.

### Result A

TN marginals solve masked completion in fewer refinement steps than local heuristics.

### Result B

PoE correction substantially improves an imperfect denoiser.

Example:

\[
\text{success}_{\mathrm{noisy}}=55\%,
\qquad
\text{success}_{\mathrm{PoE}}=85\%.
\]

### Result C

Verifier-guided remasking repairs contradictions that monotone unmasking cannot.

### Result D

TT mask policy learns a nonlocal repair rule with low rank.

Example:

\[
r=4
\]

matches oracle repair decisions almost as well as a dense model.

### Result E

The same framework transfers from Sudoku to Boolean SAT or typed expressions.

That would show the idea is not merely Sudoku-specific.

---

## 24. Next Step After the Symbolic Prototype

Once the symbolic framework is stable, bridge to language-like tasks.

The best next target is **JSON repair** or **typed expression repair**, not open-ended natural language.

Example JSON task:

```json
{
  "name": "alpha",
  "items": [
    {"id": 1, "value": true},
    {"id": 2, "value": false}
  ]
}
```

Corrupt by masking/deleting/replacing spans. Verifier:

- parse validity,
- schema validity,
- type consistency.

Denoiser:

- small masked transformer over characters or tokens.

TN/logical component:

- grammar constraints,
- schema constraints,
- bracket matching,
- field dependencies.

This is much closer to LLM use while remaining controlled.

---

## 25. Summary of the Recommended Route

The most efficient path is:

1. **Sudoku4 with brute-force exact marginals**  
   Validate the denoising/inference loop.

2. **Replace brute force by `tnreason` tensor-network contraction**  
   Validate the logical TN abstraction.

3. **Add noisy and learned denoisers**  
   Test product-of-experts correction.

4. **Add verifier-guided remasking**  
   Convert completion into repair.

5. **Add `tinyTT` mask policy**  
   Test low-rank nonlocal mask selection.

6. **Move to typed expressions or JSON repair**  
   Bridge toward real LLM-style structured generation.

7. **Only then integrate a masked diffusion LM checkpoint**  
   Use it as a denoising proposal inside the already-tested verifier/TN/remasking loop.
