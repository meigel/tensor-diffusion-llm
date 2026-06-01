"""
Brute-force marginal backend for small finite domains.

Computes exact conditional marginals over masked variables given
observed assignments, by filtering a precomputed set of all valid
solutions.

Mathematical formulation
------------------------
Given observed positions Ω with values x_Ω, the conditional marginal
for variable i ∉ Ω taking value v is:

    q_i(v) = P(x_i = v | observed)
           = Σ_{x_M \\ i} Ψ(observed, x_i=v, x_{M \\ i})
             / Σ_{x_M} Ψ(observed, x_M)

where M = {1, ..., n} \\ Ω are the masked (unobserved) positions and
Ψ is the product of all constraint indicators.

For a domain with S precomputed solutions {x^{(s)}}_{s=1}^S:

    q_i(v) = |{s : x^{(s)}_Ω = x_Ω and x^{(s)}_i = v}|
             / |{s : x^{(s)}_Ω = x_Ω}|

This is the exact oracle against which approximate tensor-network
backends are compared.
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain


class BruteForceMarginalBackend:
    """Exact marginal computation by brute-force enumeration of solutions.

    Precomputes (or caches) all valid solutions for a domain, then
    computes exact conditional marginals by counting compatible solutions.

    This is the 'oracle' backend for debugging tensor-network inference.
    Only feasible for small domains (e.g. 4×4 Sudoku with 288 solutions).
    """

    def __init__(self, domain: FiniteReasoningDomain):
        self.domain = domain
        self.n = domain.num_variables()
        self.max_d = domain.max_domain_size()
        self._solutions: Optional[np.ndarray] = None

    def _get_solutions(self) -> np.ndarray:
        """Lazily retrieve all valid solutions from the domain."""
        if self._solutions is None:
            self._solutions = self.domain.enumerate_solutions()
        return self._solutions

    def marginals(self, x_masked: np.ndarray):
        """Compute exact conditional marginals given observed assignments.

        For each masked variable i, computes:

            q_i(v) = count of compatible solutions with x_i = v
                     / count of compatible solutions

        Args:
            x_masked: Array of shape (n,) with entries in {0,...,d-1} or MASK.

        Returns:
            q:       Array of shape (n, max_d) with marginal probabilities.
                     For observed positions, q[i, x_masked[i]] = 1.0.
                     In case of contradiction, all entries are -1.0.
            n_compat: Number of compatible solutions (0 for contradiction).
            status: String — 'ok', 'contradiction', or 'all_solved'.
        """
        solutions = self._get_solutions()
        n_solutions = len(solutions)

        # Filter solutions compatible with observed positions
        compat = np.ones(n_solutions, dtype=bool)
        for i in range(self.n):
            if x_masked[i] != MASK:
                compat &= (solutions[:, i] == x_masked[i])

        n_compat = np.sum(compat)

        # No compatible solution → contradiction
        if n_compat == 0:
            q = np.full((self.n, self.max_d), -1.0, dtype=np.float64)
            return q, 0, "contradiction"

        compat_sols = solutions[compat]

        # All positions observed → deterministic delta marginals
        n_masked = np.sum(x_masked == MASK)
        if n_masked == 0:
            q = np.zeros((self.n, self.max_d), dtype=np.float64)
            for i in range(self.n):
                q[i, x_masked[i]] = 1.0
            return q, n_compat, "all_solved"

        # Compute marginals by counting
        q = np.zeros((self.n, self.max_d), dtype=np.float64)
        for i in range(self.n):
            if x_masked[i] == MASK:
                for v in range(self.max_d):
                    q[i, v] = np.mean(compat_sols[:, i] == v)
            else:
                q[i, x_masked[i]] = 1.0

        return q, n_compat, "ok"

    def size(self) -> int:
        """Return the total number of precomputed solutions."""
        return len(self._get_solutions())
