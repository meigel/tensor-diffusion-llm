"""
Finite-domain reasoning problem: base abstraction.

A FiniteReasoningDomain defines a discrete symbolic completion problem
where variables take values in finite domains A_i and constraints are
expressed as logical factors ψ_G(x_G).

The constraint satisfaction problem is governed by:

    Ψ(x) = ∏_{G ∈ G} ψ_G(x_G)

where each factor table_entry ∈ {0, 1} indicates whether a group
assignment is valid.

Masked states use the convention MASK = -1 (from tdr.MASK).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tdr import MASK


@dataclass
class VerifierDiagnostics:
    """Output of a domain verifier.

    Attributes:
        global_violation: Number of violated constraint groups.

                          V(x) = |{G ∈ G : ψ_G(x_G) = 0}|

        local_residuals:  Per-variable violation count.

                          r_i(x) = |{G ∈ G : i ∈ G and ψ_G(x_G) = 0}|

                          Shape (n,), dtype int. Used for verifier-guided
                          remasking: positions with r_i > 0 are candidates
                          for repair.
    """
    global_violation: int
    """Number of violated constraint groups."""
    local_residuals: np.ndarray
    """Per-variable violation count: how many groups containing i are violated.
       Shape (n,), dtype int."""


class FiniteReasoningDomain:
    """Base class for finite symbolic completion tasks.

    Subclasses define:
      - Variables and their domains
      - Logical constraints (via verifier and factors)
      - Solution generation for training data

    The masked state convention follows:
        x_i ∈ {MASK} ∪ {0, ..., domain_size(i) - 1}
    """

    def num_variables(self) -> int:
        """Return the number of variables n in the problem."""
        raise NotImplementedError

    def domain_size(self, i: int) -> int:
        """Return the number of distinct values d_i for variable i.

        Values are encoded as {0, 1, ..., d_i - 1}.
        """
        raise NotImplementedError

    def max_domain_size(self) -> int:
        """Return the largest domain size max_i d_i across all variables.

        Used to allocate arrays of uniform shape (n, max_d).
        """
        raise NotImplementedError

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Return a valid full assignment x of shape (n,).

        The returned array satisfies verifier(x).global_violation == 0.
        Used to generate training data for denoising.
        """
        raise NotImplementedError

    def corrupt(
        self, x: np.ndarray, mask_ratio: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a masked copy of a valid assignment (mask-only corruption).

        Each position is independently set to MASK with probability mask_ratio.

        Args:
            x: Valid assignment, shape (n,).
            mask_ratio: Probability of masking each position ∈ [0, 1].
            rng: Random number generator.

        Returns:
            x_masked: Copy of x with selected positions set to MASK.
        """
        x_masked = x.copy()
        n = self.num_variables()
        mask = rng.random(n) < mask_ratio
        x_masked[mask] = MASK
        return x_masked

    def wrong_token_corrupt(
        self, x: np.ndarray, wrong_ratio: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Replace a fraction of positions with random wrong values (no masking).

        For each position, independently with probability wrong_ratio,
        replaces the value with a uniformly random different domain value.

        Args:
            x: Valid assignment, shape (n,).
            wrong_ratio: Probability of corrupting each position ∈ [0, 1].
            rng: Random number generator.

        Returns:
            x_corrupt: Copy of x with selected positions set to wrong values.
        """
        x_corrupt = x.copy()
        n = self.num_variables()
        corrupt_mask = rng.random(n) < wrong_ratio
        for i in np.where(corrupt_mask)[0]:
            d = self.domain_size(i)
            wrong_values = [v for v in range(d) if v != x[i]]
            x_corrupt[i] = rng.choice(wrong_values)
        return x_corrupt

    def mixed_corrupt(
        self, x: np.ndarray, mask_ratio: float, wrong_ratio: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Apply both masking and wrong-token corruption.

        First masks mask_ratio fraction of positions, then corrupts
        wrong_ratio fraction of the remaining observed positions.
        A position can be both masked and corrupted (mask takes priority).

        Args:
            x: Valid assignment, shape (n,).
            mask_ratio: Probability of masking each position.
            wrong_ratio: Probability of corrupting each *observed* position.
            rng: Random number generator.

        Returns:
            x_corrupt: Copy of x with mixed corruption.
        """
        # Step 1: mask
        x_corrupt = self.corrupt(x, mask_ratio, rng)

        # Step 2: corrupt observed positions
        observed = np.where(x_corrupt != MASK)[0]
        wrong_mask = rng.random(len(observed)) < wrong_ratio
        for idx in observed[wrong_mask]:
            d = self.domain_size(idx)
            x_true = x[idx]
            wrong_values = [v for v in range(d) if v != x_true]
            x_corrupt[idx] = rng.choice(wrong_values)

        return x_corrupt

    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Return global violation and local residual information.

        Evaluates all constraint groups:

            V(x) = number of violated groups
            r_i = number of violated groups containing i

        Args:
            x: Assignment, shape (n,). May contain MASK entries.

        Returns:
            VerifierDiagnostics with global violation count and local residuals.
        """
        raise NotImplementedError

    def build_factors(self):
        """Return list of Factor objects for tensor-network or brute-force inference.

        Each Factor represents ψ_G(x_G) over a constraint group G.
        """
        raise NotImplementedError

    def enumerate_solutions(self) -> np.ndarray:
        """Return all valid full assignments as a (num_solutions, n) array.

        Subclasses with enumerable solution spaces should override this.
        The base class raises NotImplementedError.

        Returns:
            solutions: Array of shape (S, n) where S is the number of
                      valid assignments and n is num_variables().
        """
        raise NotImplementedError


@dataclass
class Factor:
    """A logical factor over a subset of variables.

    Represents a constraint indicator function:

        ψ_G(x_G) ∈ {0, 1}

    where ψ_G = 1 means the group assignment satisfies the constraint.

    Attributes:
        variables: Tuple of variable indices making up the group G.
        table:     Array of shape (d_i1, d_i2, ..., d_ik) of non-negative values.
                  1.0 means fully satisfying, 0.0 means forbidden.
                  Indexed by x[variables[0]], x[variables[1]], ...
    """
    variables: tuple[int, ...]
    table: np.ndarray
