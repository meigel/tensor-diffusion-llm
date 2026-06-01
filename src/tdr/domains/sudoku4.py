"""
4x4 Sudoku domain: 16 variables, row/col/box all-different constraints.
"""

import itertools
from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain, Factor, VerifierDiagnostics


class Sudoku4Domain(FiniteReasoningDomain):
    """4x4 Sudoku (12 constraint groups: 4 rows + 4 cols + 4 boxes).

    Variables indexed as i = 4*r + c  (r,c in {0,1,2,3}).
    Values in {0,1,2,3} internally (1,2,3,4 in standard Sudoku notation).
    """

    N = 16
    D = 4
    GRID = 4

    # Row, column, and box groups
    _GROUPS: list[tuple[int, ...]] = []

    @classmethod
    def _build_groups(cls):
        if cls._GROUPS:
            return
        groups = []
        # rows
        for r in range(4):
            groups.append(tuple(4 * r + c for c in range(4)))
        # columns
        for c in range(4):
            groups.append(tuple(4 * r + c for r in range(4)))
        # 2x2 boxes
        for br in range(2):
            for bc in range(2):
                box = tuple(
                    4 * (2 * br + r) + (2 * bc + c)
                    for r in range(2)
                    for c in range(2)
                )
                groups.append(box)
        cls._GROUPS = groups

    def __init__(self):
        self._build_groups()
        self._solutions: Optional[np.ndarray] = None

    def num_variables(self) -> int:
        return self.N

    def domain_size(self, i: int) -> int:
        return self.D

    def max_domain_size(self) -> int:
        return self.D

    # ------------------------------------------------------------------
    # Solution generator
    # ------------------------------------------------------------------
    def _generate_all_solutions(self) -> np.ndarray:
        """Enumerate all valid 4x4 Sudoku solutions via backtracking."""
        solutions = []

        def backtrack(idx, assignment):
            if idx == self.N:
                solutions.append(assignment.copy())
                return
            r, c = divmod(idx, 4)
            for v in range(self.D):
                assignment[idx] = v
                if self._is_partial_valid(assignment, idx):
                    backtrack(idx + 1, assignment)
            assignment[idx] = MASK

        backtrack(0, np.full(self.N, MASK, dtype=np.int64))
        if len(solutions) == 0:
            return np.empty((0, self.N), dtype=np.int64)
        return np.array(solutions, dtype=np.int64)

    @staticmethod
    def _is_partial_valid(x: np.ndarray, last_idx: int) -> bool:
        """Check if partial assignment up to last_idx is valid."""
        # Only check groups that are fully assigned up to last_idx
        r, c = divmod(last_idx, 4)

        # Check row
        row_vars = [4 * r + cc for cc in range(4)]
        seen = set()
        for i in row_vars:
            if x[i] == MASK:
                continue
            if x[i] in seen:
                return False
            if i <= last_idx:
                seen.add(x[i])

        # Check column
        col_vars = [4 * rr + c for rr in range(4)]
        seen = set()
        for i in col_vars:
            if x[i] == MASK:
                continue
            if x[i] in seen:
                return False
            if i <= last_idx:
                seen.add(x[i])

        # Check box
        br, bc = r // 2, c // 2
        box_vars = [4 * (2 * br + rr) + (2 * bc + cc) for rr in range(2) for cc in range(2)]
        seen = set()
        for i in box_vars:
            if x[i] == MASK:
                continue
            if x[i] in seen:
                return False
            if i <= last_idx:
                seen.add(x[i])

        return True

    def _get_solutions(self) -> np.ndarray:
        if self._solutions is None:
            self._solutions = self._generate_all_solutions()
        return self._solutions

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        solutions = self._get_solutions()
        idx = rng.integers(len(solutions))
        return solutions[idx].copy()

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------
    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Return global violation count and per-variable local residuals.

        A group is violated if any two *observed* variables share the same
        value (Option B: soft partial violation from the plan).

        Local residual r_i = number of violated groups containing variable i.
        """
        n = self.N
        global_violation = 0
        local_residuals = np.zeros(n, dtype=np.int64)

        for group in self._GROUPS:
            # Collect observed values in this group
            values_seen = {}
            violated = False
            for i in group:
                if x[i] != MASK:
                    v = x[i]
                    if v in values_seen:
                        violated = True
                        break
                    values_seen[v] = i

            if violated:
                global_violation += 1
                for i in group:
                    if x[i] != MASK:
                        local_residuals[i] += 1

        return VerifierDiagnostics(
            global_violation=int(global_violation),
            local_residuals=local_residuals,
        )

    # ------------------------------------------------------------------
    # Factors for TN / brute-force inference
    # ------------------------------------------------------------------
    def build_factors(self) -> list[Factor]:
        """Return an all-different factor for each Sudoku constraint group."""
        factors = []
        for group in self._GROUPS:
            table = self._all_different_table(len(group), self.D)
            factors.append(Factor(variables=tuple(group), table=table))
        return factors

    @staticmethod
    def _all_different_table(arity: int, d: int) -> np.ndarray:
        """Build a table that is 1.0 iff all arguments are distinct."""
        shape = (d,) * arity
        table = np.zeros(shape, dtype=np.float64)
        for assignment in itertools.product(range(d), repeat=arity):
            if len(set(assignment)) == arity:
                table[assignment] = 1.0
        return table

    # ------------------------------------------------------------------
    # Known puzzles for testing
    # ------------------------------------------------------------------
    @staticmethod
    def puzzle_easy() -> np.ndarray:
        """Return a partially filled puzzle with a unique solution."""
        # Example from the plan:
        # . 2 . .
        # . . 3 .
        # . . . 1
        # 4 . . .
        x = np.full(16, MASK, dtype=np.int64)
        x[1] = 1   # (0,1) = 2 → value 1 (0-indexed)
        x[6] = 2   # (1,2) = 3 → value 2
        x[11] = 0  # (2,3) = 1 → value 0
        x[12] = 3  # (3,0) = 4 → value 3
        return x

    @staticmethod
    def puzzle_full() -> np.ndarray:
        """Return a full valid grid for reference."""
        x = np.array([
            1, 2, 3, 0,   # row 0: 2,3,4,1
            3, 0, 1, 2,   # row 1: 4,1,2,3
            0, 3, 2, 1,   # row 2: 1,4,3,2
            2, 1, 0, 3,   # row 3: 3,2,1,4
        ], dtype=np.int64)
        return x
