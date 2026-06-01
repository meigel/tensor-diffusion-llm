"""
Tests for Sudoku4 domain: solutions, verifier, corruption.
"""

import numpy as np
import pytest

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain


class TestSudoku4Domain:
    def setup_method(self):
        self.domain = Sudoku4Domain()

    def test_num_variables(self):
        assert self.domain.num_variables() == 16

    def test_domain_size(self):
        for i in range(16):
            assert self.domain.domain_size(i) == 4

    def test_solutions_count(self):
        """4x4 Sudoku has exactly 288 unique solutions."""
        sols = self.domain.enumerate_solutions()
        assert len(sols) == 288, f"Expected 288 solutions, got {len(sols)}"
        assert sols.shape == (288, 16)

    def test_solutions_uniqueness(self):
        """All solutions should be unique."""
        sols = self.domain.enumerate_solutions()
        unique = {tuple(s) for s in sols}
        assert len(unique) == 288

    def test_solutions_valid(self):
        """Every generated solution should pass the verifier."""
        sols = self.domain.enumerate_solutions()
        for i, sol in enumerate(sols):
            assert np.all((sol >= 0) & (sol <= 3)), f"Solution {i} has invalid values"
            diag = self.domain.verifier(sol)
            assert diag.global_violation == 0, f"Solution {i} violates constraints: {sol}"

    def test_sample_solution(self):
        rng = np.random.default_rng(42)
        sol = self.domain.sample_solution(rng)
        assert sol.shape == (16,)
        assert np.all((sol >= 0) & (sol <= 3))
        diag = self.domain.verifier(sol)
        assert diag.global_violation == 0

    def test_corrupt(self):
        rng = np.random.default_rng(42)
        sol = self.domain.sample_solution(rng)
        x_masked = self.domain.corrupt(sol, mask_ratio=0.5, rng=rng)
        assert x_masked.shape == (16,)
        n_masked = np.sum(x_masked == MASK)
        assert n_masked > 0
        # Observed positions should match the solution
        observed = np.where(x_masked != MASK)[0]
        assert np.all(x_masked[observed] == sol[observed])

    def test_verifier_valid(self):
        sol = self.domain.puzzle_full()
        diag = self.domain.verifier(sol)
        assert diag.global_violation == 0
        assert np.all(diag.local_residuals == 0)

    def test_verifier_violation(self):
        x = self.domain.puzzle_full().copy()
        # Make row 0 have duplicate value 1 at positions 0 and 2
        x[0] = 0
        x[2] = 0
        diag = self.domain.verifier(x)
        assert diag.global_violation > 0
        assert diag.local_residuals[0] > 0
        assert diag.local_residuals[2] > 0

    def test_verifier_invalid_values_raises(self):
        """Verifier should reject out-of-domain values."""
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 99
        with pytest.raises(ValueError, match="Values must be MASK"):
            self.domain.verifier(x)

    def test_verifier_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            self.domain.verifier(np.ones(20, dtype=np.int64))

    def test_puzzle_easy(self):
        x = self.domain.puzzle_easy()
        assert np.sum(x == MASK) == 12
        assert np.sum(x != MASK) == 4

    def test_puzzle_easy_no_violation(self):
        """The example puzzle has no soft violations (values are distinct)."""
        x = self.domain.puzzle_easy()
        diag = self.domain.verifier(x)
        assert diag.global_violation == 0

    def test_verifier_masked_no_violation(self):
        x = np.full(16, MASK, dtype=np.int64)
        diag = self.domain.verifier(x)
        assert diag.global_violation == 0
        assert np.all(diag.local_residuals == 0)

    def test_verifier_soft_violation(self):
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 0
        x[1] = 0  # same value in row 0, col 1 → row 0 violated
        diag = self.domain.verifier(x)
        assert diag.global_violation >= 1
        assert diag.local_residuals[0] >= 1
        assert diag.local_residuals[1] >= 1

    def test_enumerate_solutions_public_method(self):
        """enumerate_solutions should be accessible via the public protocol."""
        sols = self.domain.enumerate_solutions()
        assert len(sols) == 288
