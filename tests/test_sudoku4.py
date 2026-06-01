"""
Tests for Sudoku4 domain: solutions, verifier, corruption.
"""

import numpy as np

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

    def test_solutions_generated(self):
        sols = self.domain._get_solutions()
        assert len(sols) > 0, "Should generate at least one solution"
        # Check all values in {0,1,2,3}
        assert np.all((sols >= 0) & (sols <= 3))
        # Check no solution violates constraints
        for sol in sols:
            diag = self.domain.verifier(sol)
            assert diag.global_violation == 0, f"Solution violates constraints: {sol}"

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
        # A known valid solution should have 0 violations
        sol = self.domain.puzzle_full()
        diag = self.domain.verifier(sol)
        assert diag.global_violation == 0
        assert np.all(diag.local_residuals == 0)

    def test_verifier_violation(self):
        # A row with duplicate values should be violated
        x = self.domain.puzzle_full().copy()
        # Make row 0 have two 1s
        x[0] = 0  # value 1
        x[2] = 0  # should also be value 1 in row 0 → duplicate
        diag = self.domain.verifier(x)
        assert diag.global_violation > 0
        # At least the first two positions in row 0 should have positive residuals
        assert diag.local_residuals[0] > 0
        assert diag.local_residuals[2] > 0

    def test_puzzle_easy(self):
        x = self.domain.puzzle_easy()
        assert np.sum(x == MASK) == 12  # 4 observed, 12 masked
        assert np.sum(x != MASK) == 4

    def test_verifier_masked_no_violation(self):
        """Masked positions should not cause violations."""
        x = np.full(16, MASK, dtype=np.int64)
        diag = self.domain.verifier(x)
        assert diag.global_violation == 0
        assert np.all(diag.local_residuals == 0)

    def test_verifier_soft_violation(self):
        """Two observed same values in a group → violation."""
        x = np.full(16, MASK, dtype=np.int64)
        x[0] = 0  # row 0, col 0: value 1
        x[1] = 0  # row 0, col 1: also value 1 → row 0 violated
        diag = self.domain.verifier(x)
        assert diag.global_violation >= 1
        assert diag.local_residuals[0] >= 1
        assert diag.local_residuals[1] >= 1
