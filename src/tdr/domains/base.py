"""
Finite-domain reasoning problem: base abstraction.

A FiniteReasoningDomain defines a discrete symbolic completion problem
where variables take values in finite domains and constraints are
expressed as logical factors.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tdr import MASK


@dataclass
class VerifierDiagnostics:
    """Output of a domain verifier."""
    global_violation: int
    """Number of violated constraint groups."""
    local_residuals: np.ndarray
    """Per-variable violation count: how many groups containing i are violated.
       Shape (n,), dtype int."""


class FiniteReasoningDomain:
    """Base class for finite symbolic completion tasks."""

    def num_variables(self) -> int:
        raise NotImplementedError

    def domain_size(self, i: int) -> int:
        """Return number of distinct values for variable i."""
        raise NotImplementedError

    def max_domain_size(self) -> int:
        """Return the largest domain size across all variables."""
        raise NotImplementedError

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Return a valid full assignment x of shape (n,)."""
        raise NotImplementedError

    def corrupt(
        self, x: np.ndarray, mask_ratio: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return masked assignment.

        Sets each position to MASK independently with probability mask_ratio.
        """
        x_masked = x.copy()
        n = self.num_variables()
        mask = rng.random(n) < mask_ratio
        x_masked[mask] = MASK
        return x_masked

    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Return global violation and local residual information."""
        raise NotImplementedError

    def build_factors(self):
        """Return list of Factor objects for tensor-network or brute-force inference."""
        raise NotImplementedError


@dataclass
class Factor:
    """A logical factor over a subset of variables.

    Attributes:
        variables: Tuple of variable indices.
        table:     Array of shape (d_i1, d_i2, ..., d_ik) of non-negative values.
                 1.0 means fully satisfying, 0.0 means forbidden.
    """
    variables: tuple[int, ...]
    table: np.ndarray
