"""
Marginal backends for tensor-network inference.

Defines the abstract MarginalBackend interface and provides the
ContractionMarginalBackend implementation using numpy einsum and
variable elimination.

The interface matches the plan's Section 4.2 design:

    class MarginalBackend:
        def marginals(self, domain, x_masked):
            returns (q, n_compat_or_logZ, status)
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain
from tdr.tn.factors import (
    condition_all_factors,
    contract_all_marginals,
    _infer_domain_size,
)


class MarginalBackend(ABC):
    """Abstract base for marginal computation backends."""

    @abstractmethod
    def marginals(self, x_masked: np.ndarray):
        """Compute conditional marginals given observed assignments.

        Args:
            x_masked: State array, shape (n,); MASK or domain values.

        Returns:
            q:      Array of shape (n, max_d) with marginal probabilities.
                    -1 entries if contradiction.
            aux:    Auxiliary value (number of compatible solutions for
                    brute-force, logZ for contraction backend).
            status: 'ok', 'contradiction', or 'all_solved'.
        """
        ...


class ContractionMarginalBackend(MarginalBackend):
    """Marginal computation by tensor-factor contraction (variable elimination).

    Builds factors from the domain, conditions on observed values, and
    computes exact single-variable marginals via numpy einsum.

    This is the 'exact TN' intermediate between brute-force and tnreason.
    For small domains (4x4 Sudoku) it should match brute-force marginals
    to machine precision.
    """

    def __init__(self, domain: FiniteReasoningDomain):
        self.domain = domain
        self.n = domain.num_variables()
        self.max_d = domain.max_domain_size()

    def marginals(self, x_masked: np.ndarray):
        """Compute marginals via factor contraction.

        Args:
            x_masked: State array, shape (n,); MASK or domain values.

        Returns:
            q:      Array of shape (n, max_d) of marginal probabilities.
            logZ:   Log partition function (or -inf for contradiction).
            status: 'ok', 'contradiction', or 'all_solved'.
        """
        # Build raw factors from the domain
        raw_factors = self.domain.build_factors()

        # Condition on observed values
        conditioned = condition_all_factors(raw_factors, x_masked)

        # Remove scalar factors (all variables observed in that factor)
        # to keep the elimination efficient
        non_scalar = [f for f in conditioned if len(f.variables) > 0]

        if not non_scalar:
            # All factors are scalars — everything is observed
            # Check for contradiction: any zero scalar?
            contradiction = False
            for f in conditioned:
                if len(f.variables) == 0 and f.table.size > 0 and f.table.ravel()[0] == 0.0:
                    contradiction = True
                    break

            if contradiction:
                q = np.full((self.n, self.max_d), -1.0, dtype=np.float64)
                return q, -np.inf, "contradiction"

            # All observed and consistent — delta marginals
            q = np.zeros((self.n, self.max_d), dtype=np.float64)
            for i in range(self.n):
                if x_masked[i] != MASK:
                    q[i, x_masked[i]] = 1.0
            return q, 0.0, "all_solved"

        # Compute all marginals via variable elimination
        q, logZ, status = contract_all_marginals(
            non_scalar, self.n, self.max_d,
        )

        if status == "contradiction":
            return q, logZ, status

        # Override observed positions with delta distributions
        for i in range(self.n):
            if x_masked[i] != MASK:
                q[i, :] = 0.0
                q[i, x_masked[i]] = 1.0

        return q, logZ, status
