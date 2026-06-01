"""
Denoisers for masked diffusion.

Provides:
- RandomDenoiser: uniform random filling.
- LocalSudokuDenoiser: row/col/box local allowed-values heuristic.
- TNMarginalDenoiser: wraps a marginal backend (e.g. brute force).
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain
from tdr.tn.brute_force_backend import BruteForceMarginalBackend


class RandomDenoiser:
    """Denoiser that predicts uniformly over feasible values."""

    def __init__(self, domain: FiniteReasoningDomain):
        self.domain = domain
        self.n = domain.num_variables()
        self.d = domain.max_domain_size()

    def predict(self, x_masked: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return uniform distribution over all values for masked positions."""
        q = np.full((self.n, self.d), 1.0 / self.d, dtype=np.float64)
        for i in range(self.n):
            if x_masked[i] != MASK:
                q[i, :] = 0.0
                q[i, x_masked[i]] = 1.0
        return q


class LocalSudokuDenoiser:
    """Local heuristic denoiser for Sudoku.

    For each masked position, predict uniform over values that do not
    immediately conflict with observed neighbours in the same row,
    column, or box.  Does NOT reason globally.
    """

    def __init__(self, domain: FiniteReasoningDomain):
        self.domain = domain
        self.n = domain.num_variables()
        self.d = domain.max_domain_size()
        # Precompute group membership for each variable
        self._groups_of: list[list[tuple[int, ...]]] = [[] for _ in range(self.n)]
        for group in domain._GROUPS:  # type: ignore
            for i in group:
                self._groups_of[i].append(group)

    def predict(self, x_masked: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        q = np.zeros((self.n, self.d), dtype=np.float64)

        for i in range(self.n):
            if x_masked[i] != MASK:
                q[i, x_masked[i]] = 1.0
                continue

            # Collect values already observed in groups containing i
            forbidden = set()
            for group in self._groups_of[i]:
                for j in group:
                    if j != i and x_masked[j] != MASK:
                        forbidden.add(x_masked[j])

            allowed = [v for v in range(self.d) if v not in forbidden]
            if allowed:
                for v in allowed:
                    q[i, v] = 1.0 / len(allowed)
            else:
                # All values are locally forbidden -> fallback to uniform
                q[i, :] = 1.0 / self.d

        return q


class TNMarginalDenoiser:
    """Denoiser that uses exact logical marginals from a marginal backend.

    Wraps BruteForceMarginalBackend (or future TNReasonBackend).
    """

    def __init__(self, backend: BruteForceMarginalBackend):
        self.backend = backend
        self.n = backend.n
        self.d = backend.max_d

    def predict(self, x_masked: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        q, n_compat, status = self.backend.marginals(x_masked)
        if status == "contradiction":
            # No compatible solutions: return uniform as fallback
            q = np.full((self.n, self.d), 1.0 / self.d, dtype=np.float64)
        return q
