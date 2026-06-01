"""
Denoisers for masked diffusion.

Provides proposal distributions p_i(v) for each masked variable i.
These are the 'noise predictor' side of the masked diffusion process.

Mathematical setting
--------------------
At each diffusion step k, given a masked state x^{(k)}, a denoiser
produces a distribution over values for each masked position:

    p_i(v) = Proposal(x_i = v | x^{(k)}_{observed})

Denoisers range from trivial (uniform random) through heuristic
(local allowed values) to exact (TN marginals).

Denoiser types
--------------
1. RandomDenoiser:      p_i(v) = 1/d (uniform)
2. LocalSudokuDenoiser: p_i(v) ∝ 1{v not locally forbidden}
3. TNMarginalDenoiser:  p_i(v) = q_i(v) (exact TN marginal)
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain
from tdr.tn.brute_force_backend import BruteForceMarginalBackend


class RandomDenoiser:
    """Denoiser that predicts uniformly over all domain values.

    p_i(v) = 1/d  for all masked positions i and all values v.

    Acts as the weakest baseline — random guessing without constraints.
    """

    def __init__(self, domain: FiniteReasoningDomain):
        self.domain = domain
        self.n = domain.num_variables()
        self.d = domain.max_domain_size()

    def predict(self, x_masked: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return uniform distribution over all values.

        For observed positions, returns a delta distribution.
        For masked positions, returns uniform 1/d over all d values.

        Args:
            x_masked: State array, shape (n,); entries in {0,...,d-1} or MASK.
            rng:      Ignored (included for interface compatibility).

        Returns:
            q: Array of shape (n, d) of probability distributions.
        """
        q = np.full((self.n, self.d), 1.0 / self.d, dtype=np.float64)
        for i in range(self.n):
            if x_masked[i] != MASK:
                q[i, :] = 0.0
                q[i, x_masked[i]] = 1.0
        return q


class LocalSudokuDenoiser:
    """Local heuristic denoiser using row/col/box forbidden values.

    For each masked position i, identifies values that appear in
    already-observed positions of the same row, column, or 2×2 box.
    The proposal is uniform over the remaining (allowed) values.

    This heuristic reasons only locally and does NOT capture global
    constraint interactions. It serves as a structured baseline
    between random guessing and exact TN marginals.
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
        """Compute locally-allowed proposal distribution.

        For each masked position i:
            1. Collect all values that appear in observed positions
               sharing a row, column, or box with i (the 'forbidden set').
            2. Distribute probability uniformly over remaining values.
            3. Fallback to uniform if all values are locally forbidden.

        Args:
            x_masked: State array, shape (n,).
            rng:      Ignored.

        Returns:
            q: Array of shape (n, d) of probability distributions.
        """
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
                # All values are locally forbidden — fallback to uniform
                q[i, :] = 1.0 / self.d

        return q


class TNMarginalDenoiser:
    """Denoiser using exact logical marginals from a backend.

    p_i(v) = q_i(v) = P_Ψ(x_i = v | x_{observed})

    where q_i(v) comes from a BruteForceMarginalBackend (or future
    tensor-network backend). This is the exact oracle denoiser.
    """

    def __init__(self, backend: BruteForceMarginalBackend):
        self.backend = backend
        self.n = backend.n
        self.d = backend.max_d

    def predict(self, x_masked: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return exact TN marginal distributions.

        Delegates to the marginal backend. If the observed state is
        a contradiction (no compatible solutions), falls back to
        uniform distribution.

        Args:
            x_masked: State array, shape (n,).
            rng:      Ignored.

        Returns:
            q: Array of shape (n, d) of marginal probability distributions.
        """
        q, n_compat, status = self.backend.marginals(x_masked)
        if status == "contradiction":
            q = np.full((self.n, self.d), 1.0 / self.d, dtype=np.float64)
        return q
