"""
Tests for brute-force marginal backend.
"""

import numpy as np

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.brute_force_backend import BruteForceMarginalBackend


class TestBruteForceMarginalBackend:
    def setup_method(self):
        self.domain = Sudoku4Domain()
        self.backend = BruteForceMarginalBackend(self.domain)

    def test_solutions_loaded(self):
        assert self.backend.size() > 0

    def test_marginals_full_mask(self):
        """All masked: marginals should be uniform over valid values."""
        x_masked = np.full(16, MASK, dtype=np.int64)
        q, n_compat, status = self.backend.marginals(x_masked)
        assert status == "ok"
        assert n_compat == self.backend.size()
        # Each marginal should sum to 1
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} does not sum to 1"

    def test_marginals_fully_observed(self):
        """Fully observed valid solution: marginals should be delta."""
        sol = self.domain.sample_solution(np.random.default_rng(42))
        q, n_compat, status = self.backend.marginals(sol)
        assert status == "all_solved"
        assert n_compat >= 1
        for i in range(16):
            assert q[i, sol[i]] == 1.0
            assert np.sum(q[i]) == 1.0

    def test_marginals_contradiction(self):
        """Inconsistent assignment → contradiction."""
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 0
        x[1] = 0  # same value in same row → impossible for 4x4 Sudoku
        # This might still have solutions since row 0 could have 1,1,*,*
        # Actually, rows must have all distinct values, so two 1s in same row IS impossible
        q, n_compat, status = self.backend.marginals(x)
        assert status == "contradiction"
        assert n_compat == 0

    def test_marginals_partial(self):
        """Partially observed: marginals should be consistent."""
        rng = np.random.default_rng(42)
        sol = self.domain.sample_solution(rng)
        x_masked = self.domain.corrupt(sol, mask_ratio=0.5, rng=rng)

        q, n_compat, status = self.backend.marginals(x_masked)
        assert status in ("ok", "all_solved")
        assert n_compat >= 1

        # Check the observed positions have delta marginals
        for i in range(16):
            if x_masked[i] != MASK:
                assert q[i, x_masked[i]] == 1.0
                assert np.sum(q[i]) == 1.0

        # All marginals should sum to 1 (valid probability distributions)
        for i in range(16):
            assert np.isclose(q[i].sum(), 1.0), f"Marginal {i} sum={q[i].sum()}"

    def test_marginals_known_puzzle(self):
        """Test against the known easy puzzle."""
        x = self.domain.puzzle_easy()
        q, n_compat, status = self.backend.marginals(x)
        assert status == "ok"
        assert n_compat >= 1
        # Observed positions should have delta
        for i in range(16):
            if x[i] != MASK:
                assert q[i, x[i]] == 1.0
