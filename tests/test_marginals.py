"""
Tests for brute-force and contraction marginal backends.

Verifies:
  - Brute-force backend produces correct marginals
  - Contraction (variable elimination) backend produces exact marginals
  - Both backends agree to machine precision: ‖q^C − q^BF‖_∞ < 1e-14
"""

import numpy as np
import pytest

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.brute_force_backend import BruteForceMarginalBackend
from tdr.tn.marginals import ContractionMarginalBackend


# Shared domain instance
DOMAIN = Sudoku4Domain()
N_SOLUTIONS = 288


class TestBruteForceMarginalBackend:
    """Tests specific to the brute-force (solution-filtering) backend."""

    def setup_method(self):
        self.backend = BruteForceMarginalBackend(DOMAIN)

    def test_solutions_count(self):
        assert self.backend.size() == N_SOLUTIONS

    def test_marginals_full_mask(self):
        """All-masked state: marginals should be probability distributions."""
        x_masked = np.full(16, MASK, dtype=np.int64)
        q, n_compat, status = self.backend.marginals(x_masked)
        assert status == "ok"
        assert n_compat == N_SOLUTIONS
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} sum={q[i].sum()}"
            assert np.all(q[i] >= 0.0), f"Marginal {i} has negative entries"

    def test_marginals_fully_observed_valid(self):
        """Fully observed valid solution: delta marginals."""
        rng = np.random.default_rng(42)
        sol = DOMAIN.sample_solution(rng)
        q, n_compat, status = self.backend.marginals(sol)
        assert status == "all_solved"
        assert n_compat >= 1
        for i in range(16):
            assert q[i, sol[i]] == 1.0
            assert np.isclose(q[i].sum(), 1.0)

    def test_marginals_contradiction(self):
        """Inconsistent assignment → contradiction with 0 compatible solutions."""
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 0
        x[1] = 0  # same value in row 0 → impossible
        q, n_compat, status = self.backend.marginals(x)
        assert status == "contradiction"
        assert n_compat == 0
        assert q.shape == (16, 4)
        assert np.all(q == -1.0)

    def test_marginals_partial_consistent(self):
        """Partially observed: marginals should be consistent and sum to 1."""
        rng = np.random.default_rng(42)
        sol = DOMAIN.sample_solution(rng)
        x_masked = DOMAIN.corrupt(sol, mask_ratio=0.5, rng=rng)

        q, n_compat, status = self.backend.marginals(x_masked)
        assert status in ("ok", "all_solved")
        assert n_compat >= 1

        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} sum={q[i].sum()}"
            if x_masked[i] != MASK:
                assert q[i, x_masked[i]] == 1.0

    def test_marginals_known_puzzle_unique(self):
        """puzzle_easy has exactly 1 compatible solution."""
        x = DOMAIN.puzzle_easy()
        q, n_compat, status = self.backend.marginals(x)
        assert status == "ok"
        assert n_compat == 1, f"Expected unique solution, got {n_compat}"
        for i in range(16):
            assert np.all((q[i] == 0.0) | (q[i] == 1.0))
            assert np.isclose(q[i].sum(), 1.0)

    def test_marginals_zero_mask_ratio(self):
        """Fully observed (no masks): should be all_solved."""
        rng = np.random.default_rng(42)
        sol = DOMAIN.sample_solution(rng)
        q, n_compat, status = self.backend.marginals(sol)
        assert status == "all_solved"
        for i in range(16):
            assert q[i, sol[i]] == 1.0

    def test_backend_via_public_protocol(self):
        """Backend should use the public enumerate_solutions method."""
        backend = BruteForceMarginalBackend(DOMAIN)
        assert backend.size() == N_SOLUTIONS


class TestContractionMarginalBackend:
    """Tests for the variable-elimination contraction backend."""

    def setup_method(self):
        self.backend = ContractionMarginalBackend(DOMAIN)

    def test_marginals_full_mask(self):
        """All-masked: marginals should be proper distributions."""
        x_masked = np.full(16, MASK, dtype=np.int64)
        q, logZ, status = self.backend.marginals(x_masked)
        assert status == "ok"
        assert np.isfinite(logZ)
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} sum={q[i].sum()}"
            assert np.all(q[i] >= 0.0)

    def test_marginals_fully_observed_valid(self):
        """Full valid solution: delta marginals."""
        rng = np.random.default_rng(42)
        sol = DOMAIN.sample_solution(rng)
        q, logZ, status = self.backend.marginals(sol)
        assert status == "all_solved"
        for i in range(16):
            assert q[i, sol[i]] == 1.0
            assert np.isclose(q[i].sum(), 1.0)

    def test_marginals_contradiction(self):
        """Inconsistent observed values → contradiction."""
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 0
        x[1] = 0
        q, logZ, status = self.backend.marginals(x)
        assert status == "contradiction"
        assert np.all(q == -1.0)

    def test_marginals_partial_consistent(self):
        """Partially observed: consistent marginals."""
        rng = np.random.default_rng(42)
        sol = DOMAIN.sample_solution(rng)
        x_masked = DOMAIN.corrupt(sol, mask_ratio=0.5, rng=rng)
        q, logZ, status = self.backend.marginals(x_masked)
        assert status in ("ok", "all_solved")
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0)
            if x_masked[i] != MASK:
                assert q[i, x_masked[i]] == 1.0

    def test_puzzle_easy(self):
        """puzzle_easy should yield deterministic marginals via contraction."""
        x = DOMAIN.puzzle_easy()
        q, logZ, status = self.backend.marginals(x)
        assert status == "ok"
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0)
            # With a unique solution, all marginals should be 0/1
            assert np.all((q[i] == 0.0) | (q[i] == 1.0))


class TestBackendAgreement:
    """Verify that contraction matches brute-force to machine precision.

    Criterion from the plan (Section 4.3):
        ‖q^C − q^BF‖_∞ < 1e-10
    """

    def setup_method(self):
        self.bf = BruteForceMarginalBackend(DOMAIN)
        self.ct = ContractionMarginalBackend(DOMAIN)

    def _check_agreement(self, x_masked, label=""):
        q_bf, _, s_bf = self.bf.marginals(x_masked)
        q_ct, _, s_ct = self.ct.marginals(x_masked)
        assert s_bf == s_ct, (
            f"Status mismatch: BF={s_bf}, CT={s_ct} ({label})"
        )
        max_diff = np.max(np.abs(q_bf - q_ct))
        assert max_diff < 1e-14, (
            f"‖q^C − q^BF‖_∞ = {max_diff:.2e} > 1e-14 ({label})"
        )
        return max_diff

    def test_all_masked(self):
        self._check_agreement(np.full(16, MASK, dtype=np.int64), "all_masked")

    def test_full_valid(self):
        rng = np.random.default_rng(0)
        sol = DOMAIN.sample_solution(rng)
        self._check_agreement(sol, "full_valid")

    def test_puzzle_easy(self):
        self._check_agreement(DOMAIN.puzzle_easy(), "puzzle_easy")

    def test_contradiction(self):
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 0
        x[1] = 0
        self._check_agreement(x, "contradiction")

    def test_random_mask_ratios(self):
        """Agreement across mask ratios 0.25, 0.50, 0.75, 0.90."""
        rng = np.random.default_rng(42)
        sol = DOMAIN.sample_solution(rng)
        for ratio in [0.25, 0.50, 0.75, 0.90]:
            x_masked = DOMAIN.corrupt(sol, ratio, rng)
            self._check_agreement(x_masked, f"mask_ratio={ratio}")

    def test_many_random_seeds(self):
        """Statistical test: 50 random seeds, all must agree to 1e-14."""
        max_diffs = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sol = DOMAIN.sample_solution(rng)
            x_masked = DOMAIN.corrupt(sol, 0.5, rng)
            md = self._check_agreement(x_masked, f"seed={seed}")
            max_diffs.append(md)
        assert np.max(max_diffs) < 1e-14, (
            f"Worst-case diff across 50 seeds: {np.max(max_diffs):.2e}"
        )

    def test_many_different_solutions(self):
        """50 different underlying solutions, all must agree."""
        max_diffs = []
        rng_mask = np.random.default_rng(999)
        for seed in range(50):
            rng = np.random.default_rng(seed)
            sol = DOMAIN.sample_solution(rng)
            x_masked = DOMAIN.corrupt(sol, 0.5, rng_mask)
            md = self._check_agreement(x_masked, f"sol_seed={seed}")
            max_diffs.append(md)
        assert np.max(max_diffs) < 1e-14
