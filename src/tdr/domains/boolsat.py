"""
Planted sparse k-SAT domain for controlled difficulty scaling.

Generates random satisfiable k-CNF formulas with a known planted solution.
This is the harder controlled domain recommended by Codex 5.5 for
demonstrating TN-guided repair.

Variable encoding
-----------------
n binary variables, values {0, 1} (False, True).
Clauses are k-CNF with a fixed clause width k (default 3).

Constraint factors
------------------
Each clause C = (ℓ_1 ∨ ℓ_2 ∨ ... ∨ ℓ_k) is encoded as a factor:

    ψ_C(x_{∂C}) = 0  iff  all literals are False
                = 1  otherwise

Verifier
--------
Global violation V(x) = number of unsatisfied clauses.
Local residual r_i(x) = number of clauses containing literal i that are violated.

Difficulty control
------------------
- n_vars: number of propositional variables
- n_clauses: number of clauses (control clause/variable ratio)
- k: clause width (default 3)
- Higher clause/variable ratio → harder problem
"""

import numpy as np
from typing import Optional

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain, Factor, VerifierDiagnostics


class BoolSatDomain(FiniteReasoningDomain):
    """Planted sparse k-SAT domain.

    Generates a random satisfiable k-CNF formula with a known planted
    solution, then provides verification and factor-based inference.

    Attributes:
        n_vars:    Number of Boolean variables.
        n_clauses: Number of clauses.
        k:         Clause width (default 3).
    """

    def __init__(self, n_vars: int = 20, n_clauses: int = 60, k: int = 3,
                 rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()
        self._n = n_vars
        self._k = k
        self._planted = rng.integers(0, 2, size=n_vars, dtype=np.int64)
        self._clauses = self._generate_clauses(n_clauses, self._planted, rng)
        self._n_clauses = len(self._clauses)

    @staticmethod
    def _generate_clauses(n_clauses: int, planted: np.ndarray,
                          rng: np.random.Generator, k: int = 3) -> list:
        """Generate random k-CNF clauses satisfied by planted assignment."""
        n_vars = len(planted)
        clauses = []
        # To avoid duplicates, use a set of frozensets
        clause_set = set()

        while len(clauses) < n_clauses:
            # Pick k distinct variables
            vars_idx = rng.choice(n_vars, size=k, replace=False)

            # Normalize to a canonical key
            key = tuple(sorted(vars_idx))
            if key in clause_set:
                continue
            clause_set.add(key)

            # For each variable, pick sign such that the clause is satisfied
            # when at least one literal is True
            # Strategy: randomly choose signs, but ensure at least one
            # literal evaluates to True under the planted assignment
            signs = rng.integers(0, 2, size=k, dtype=np.int64)

            # Check if at least one literal is True under planted assignment
            # literal_i = (sign_i == 0) means positive literal: x_i
            # literal_i = (sign_i == 1) means negative literal: ~x_i
            # literal is True if (x_i == 1 and literal is positive)
            #                  or (x_i == 0 and literal is negative)
            # i.e., (planted[vars_idx[i]] == 1 and signs[i] == 0)
            #    or (planted[vars_idx[i]] == 0 and signs[i] == 1)
            # i.e., planted[vars_idx[i]] != signs[i]
            any_true = np.any(planted[vars_idx] != signs)

            if not any_true:
                # Flip a random sign to make it satisfied
                flip_idx = rng.integers(k)
                signs[flip_idx] = 1 - signs[flip_idx]

            clauses.append({
                "vars": tuple(vars_idx),
                "signs": tuple(signs),
            })

        return clauses

    def num_variables(self) -> int:
        return self._n

    def domain_size(self, i: int) -> int:
        return 2  # Boolean

    def max_domain_size(self) -> int:
        return 2

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Return the planted solution."""
        return self._planted.copy()

    def enumerate_solutions(self) -> np.ndarray:
        """Return just the planted solution (single known solution).

        For SAT with many solutions, we only have the planted one.
        Brute-force marginal backend will be very limited with this —
        use the contraction backend instead.
        """
        return self._planted.copy().reshape(1, -1)

    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Count unsatisfied clauses and local residuals.

        A clause is violated iff all its literals are False:

            violated(C) = AND_{i in C} (literal_i(x) is False)

        where literal_i(x) = x_i if sign_i=0, else 1-x_i.

        Global violation: number of unsatisfied clauses.
        Local residual: number of unsatisfied clauses containing each variable.
        """
        if x.shape != (self._n,):
            raise ValueError(f"Expected shape ({self._n},), got {x.shape}")

        n = self._n
        global_violation = 0
        local_residuals = np.zeros(n, dtype=np.int64)

        for clause in self._clauses:
            vars_idx = clause["vars"]
            signs = clause["signs"]

            # Check if clause is satisfied
            satisfied = False
            for idx, sign in zip(vars_idx, signs):
                if x[idx] == MASK:
                    continue
                literal_true = (x[idx] == 1 and sign == 0) or \
                               (x[idx] == 0 and sign == 1)
                if literal_true:
                    satisfied = True
                    break

            if not satisfied and not np.any(x[list(vars_idx)] == MASK):
                # All observed literals are False → clause violated
                global_violation += 1
                for idx in vars_idx:
                    if x[idx] != MASK:
                        local_residuals[idx] += 1

        return VerifierDiagnostics(
            global_violation=int(global_violation),
            local_residuals=local_residuals,
        )

    def build_factors(self) -> list[Factor]:
        """Build clause satisfaction factors.

        Each clause yields a factor of shape (2, 2, ..., 2) with
        entry 0 iff all literals are False under that assignment.
        """
        factors = []
        for clause in self._clauses:
            vars_idx = clause["vars"]
            signs = clause["signs"]
            table = np.ones((2,) * len(vars_idx), dtype=np.float64)

            # The only disallowed assignment is when all literals are False
            # literal_i False when (sign_i=0 and x_i=0) or (sign_i=1 and x_i=1)
            # i.e., x_i == sign_i
            disallowed = tuple(signs)
            table[disallowed] = 0.0

            factors.append(Factor(variables=vars_idx, table=table))

        return factors

    @property
    def n_clauses(self) -> int:
        return self._n_clauses

    @property
    def clause_list(self) -> list:
        """Return human-readable clause strings."""
        strings = []
        for c in self._clauses:
            lit_strs = []
            for v, s in zip(c["vars"], c["signs"]):
                if s == 0:
                    lit_strs.append(f"x{v}")
                else:
                    lit_strs.append(f"~x{v}")
            strings.append("(" + " ∨ ".join(lit_strs) + ")")
        return strings
