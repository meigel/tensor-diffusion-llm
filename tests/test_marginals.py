"""
Tests for brute-force marginal backend.
"""

import numpy as np
import pytest

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.brute_force_backend import BruteForceMarginalBackend


class TestBruteForceMarginalBackend:
    def setup_method(self):
        self.domain = Sudoku4Domain()
        self.backend = BruteForceMarginalBackend(self.domain)
        self.n_solutions = 288

    def test_solutions_count(self):
        assert self.backend.size() == self.n_solutions

    def test_marginals_full_mask(self):
        """All-masked state: marginals should be probability distributions."""
        x_masked = np.full(16, MASK, dtype=np.int64)
        q, n_compat, status = self.backend.marginals(x_masked)
        assert status == "ok"
        assert n_compat == self.n_solutions
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} sum={q[i].sum()}"
            assert np.all(q[i] >= 0.0), f"Marginal {i} has negative entries"

    def test_marginals_fully_observed_valid(self):
        """Fully observed valid solution: delta marginals."""
        rng = np.random.default_rng(42)
        sol = self.domain.sample_solution(rng)
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
        sol = self.domain.sample_solution(rng)
        x_masked = self.domain.corrupt(sol, mask_ratio=0.5, rng=rng)

        q, n_compat, status = self.backend.marginals(x_masked)
        assert status in ("ok", "all_solved")
        assert n_compat >= 1

        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} sum={q[i].sum()}"
            if x_masked[i] != MASK:
                assert q[i, x_masked[i]] == 1.0

    def test_marginals_known_puzzle_unique(self):
        """puzzle_easy has exactly 1 compatible solution."""
        x = self.domain.puzzle_easy()
        q, n_compat, status = self.backend.marginals(x)
        assert status == "ok"
        assert n_compat == 1, f"Expected unique solution, got {n_compat}"
        # With exactly 1 solution, all marginals should be 0/1
        for i in range(16):
            assert np.all((q[i] == 0.0) | (q[i] == 1.0))
            assert np.isclose(q[i].sum(), 1.0)

    def test_marginals_zero_mask_ratio(self):
        """Fully observed (no masks): should be all_solved."""
        rng = np.random.default_rng(42)
        sol = self.domain.sample_solution(rng)
        q, n_compat, status = self.backend.marginals(sol)
        assert status == "all_solved"
        for i in range(16):
            assert q[i, sol[i]] == 1.0

    def test_backend_via_public_protocol(self):
        """Backend should use the public enumerate_solutions method."""
        backend = BruteForceMarginalBackend(self.domain)
        assert backend.size() == self.n_solutions
