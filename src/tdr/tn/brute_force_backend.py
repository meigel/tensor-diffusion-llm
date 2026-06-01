"""
Brute-force marginal backend for small finite domains.

Precomputes all valid solutions for a domain, then computes exact
conditional marginals given observed assignments by filtering compatible
solutions.
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain


class BruteForceMarginalBackend:
    """Exact marginal computation by brute-force enumeration of solutions.

    This is the 'oracle' backend for debugging tensor-network inference.
    Only feasible for small domains (e.g. 4x4 Sudoku with ~288 solutions).
    """

    def __init__(self, domain: FiniteReasoningDomain):
        self.domain = domain
        self.n = domain.num_variables()
        self.max_d = domain.max_domain_size()
        self._solutions: Optional[np.ndarray] = None

    def _get_solutions(self) -> np.ndarray:
        if self._solutions is None:
            self._solutions = self.domain.enumerate_solutions()
        return self._solutions

    def marginals(self, x_masked: np.ndarray):
        """Compute exact marginals given observed assignments.

        Args:
            x_masked: Array of shape (n,) with entries in {0,...,d-1} or MASK.

        Returns:
            q:      Array of shape (n, max_d) with marginal probabilities.
                    Positions with no compatible solutions have q[i] = -1.
            n_compat: Number of compatible solutions.
            status: 'ok', 'contradiction', or 'all_solved' (if no masks left).
        """
        solutions = self._get_solutions()
        n_solutions = len(solutions)

        # Filter compatible solutions
        compat = np.ones(n_solutions, dtype=bool)
        for i in range(self.n):
            if x_masked[i] != MASK:
                compat &= (solutions[:, i] == x_masked[i])

        n_compat = np.sum(compat)

        if n_compat == 0:
            q = np.full((self.n, self.max_d), -1.0, dtype=np.float64)
            return q, 0, "contradiction"

        compat_sols = solutions[compat]

        # Check if all positions are observed
        n_masked = np.sum(x_masked == MASK)
        if n_masked == 0:
            q = np.zeros((self.n, self.max_d), dtype=np.float64)
            for i in range(self.n):
                q[i, x_masked[i]] = 1.0
            return q, n_compat, "all_solved"

        # Compute marginals
        q = np.zeros((self.n, self.max_d), dtype=np.float64)
        for i in range(self.n):
            if x_masked[i] == MASK:
                for v in range(self.max_d):
                    q[i, v] = np.mean(compat_sols[:, i] == v)
            else:
                q[i, x_masked[i]] = 1.0

        return q, n_compat, "ok"

    def size(self) -> int:
        """Return number of precomputed solutions."""
        return len(self._get_solutions())
