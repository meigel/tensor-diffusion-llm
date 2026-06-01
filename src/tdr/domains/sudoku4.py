"""
4x4 Sudoku domain: 16 variables, row/col/box all-different constraints.

Variable encoding
-----------------
Variables are indexed as i = 4·r + c with r, c ∈ {0, 1, 2, 3}.
Values are {0, 1, 2, 3} internally, representing {1, 2, 3, 4} in
standard Sudoku notation.

Constraint groups
-----------------
There are 12 all-different constraint groups G:

  • 4 rows:       each row must contain {0, 1, 2, 3} exactly once
  • 4 columns:    each column must contain {0, 1, 2, 3} exactly once
  • 4 boxes:      each 2×2 box must contain {0, 1, 2, 3} exactly once

Each group factor ψ_G(x_G) = 1 iff all elements of x_G are distinct.

Verifier (Option B — soft partial violation)
---------------------------------------------
A group is violated if any two *observed* (non-MASK) variables share
the same value. Masked variables are ignored:

    V(x) = Σ_{G ∈ G} 1{∃ i,j ∈ G : x_i = x_j ≠ MASK}

Local residual:
    r_i(x) = |{G ∈ G : i ∈ G and G is violated}|

Solutions
---------
The full solution set contains exactly 288 valid 4×4 Sudoku completions
(enumerated by depth-first backtracking with partial validity checking).
"""

import itertools
from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import FiniteReasoningDomain, Factor, VerifierDiagnostics


class Sudoku4Domain(FiniteReasoningDomain):
    """4x4 Sudoku domain with 12 all-different constraint groups.

    The state space has n = 16 variables, each taking values in {0,1,2,3}.
    A complete valid assignment satisfies all row, column, and 2×2 box
    all-different constraints simultaneously.
    """

    N = 16
    D = 4
    GRID = 4

    # Row, column, and box groups
    _GROUPS: list[tuple[int, ...]] = []

    @classmethod
    def _build_groups(cls):
        """Build the 12 constraint groups (4 rows + 4 columns + 4 boxes)."""
        if cls._GROUPS:
            return
        groups = []
        # rows: each row is {4r, 4r+1, 4r+2, 4r+3}
        for r in range(4):
            groups.append(tuple(4 * r + c for c in range(4)))
        # columns: each column is {r, r+4, r+8, r+12}
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
        """Enumerate all 288 valid 4x4 Sudoku solutions via DFS backtracking.

        The search starts from an all-MASK state and fills positions in
        index order (0 through 15). At each step _is_partial_valid prunes
        branches that cannot lead to a valid solution.

        Returns:
            solutions: Array of shape (288, 16) of all valid completions.
        """
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
        """Check whether a partial assignment up to position last_idx is valid.

        Examines the row, column, and box containing last_idx. Only positions
        ≤ last_idx are considered; future positions (still MASK) are ignored.

        Args:
            x: Partial assignment, shape (16,).
            last_idx: The index last assigned (0-indexed).

        Returns:
            True if the assignment satisfies all constraints among the
            assigned positions within each constraint group.
        """
        r, c = divmod(last_idx, 4)

        # Check row: no duplicate values in row r among positions ≤ last_idx
        row_vars = [4 * r + cc for cc in range(4)]
        seen = set()
        for i in row_vars:
            if x[i] == MASK:
                continue
            if x[i] in seen:
                return False
            if i <= last_idx:
                seen.add(x[i])

        # Check column: no duplicate values in column c
        col_vars = [4 * rr + c for rr in range(4)]
        seen = set()
        for i in col_vars:
            if x[i] == MASK:
                continue
            if x[i] in seen:
                return False
            if i <= last_idx:
                seen.add(x[i])

        # Check box: no duplicate values in the 2×2 box
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
        """Lazily generate and cache all solutions."""
        if self._solutions is None:
            self._solutions = self._generate_all_solutions()
        return self._solutions

    def sample_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Return a uniformly random valid solution."""
        solutions = self._get_solutions()
        idx = rng.integers(len(solutions))
        return solutions[idx].copy()

    def enumerate_solutions(self) -> np.ndarray:
        """Return all 288 valid solutions."""
        return self._get_solutions().copy()

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------
    def verifier(self, x: np.ndarray) -> VerifierDiagnostics:
        """Evaluate constraint violations (Option B — soft partial violation).

        A group G is violated if any two *observed* (non-MASK) variables
        in G share the same value. Masked variables are ignored.

        Global violation:
            V(x) = Σ_{G ∈ G} 1{∃ i,j ∈ G : x_i = x_j ≠ MASK}

        Local residual:
            r_i(x) = |{G ∈ G : i ∈ G and G is violated}|

        Args:
            x: Assignment, shape (16,). Entries in {MASK, 0, 1, 2, 3}.

        Returns:
            VerifierDiagnostics with counts.

        Raises:
            ValueError: If x has wrong shape or values outside {MASK, 0, 1, 2, 3}.
        """
        if x.shape != (self.N,):
            raise ValueError(
                f"Expected shape ({self.N},), got {x.shape}"
            )
        if not np.all((x == MASK) | ((x >= 0) & (x < self.D))):
            raise ValueError(
                f"Values must be MASK={MASK} or in [0, {self.D - 1}], "
                f"got {set(x[x != MASK].tolist()) - set(range(self.D))}"
            )

        n = self.N
        global_violation = 0
        local_residuals = np.zeros(n, dtype=np.int64)

        for group in self._GROUPS:
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
        """Return an all-different factor for each of the 12 constraint groups.

        Each factor table has shape (4, 4, 4, 4) with entry 1.0 iff all
        four values are pairwise distinct.
        """
        factors = []
        for group in self._GROUPS:
            table = self._all_different_table(len(group), self.D)
            factors.append(Factor(variables=tuple(group), table=table))
        return factors

    @staticmethod
    def _all_different_table(arity: int, d: int) -> np.ndarray:
        """Build a factor table that is 1.0 iff all arguments are distinct.

        ψ(x_1, ..., x_k) = 1{x_i ≠ x_j for all i ≠ j}

        The resulting table has shape (d,) × arity.
        """
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
        """Return a partially filled 4×4 Sudoku puzzle with exactly one solution.

        Puzzle (standard notation):
            . 2 . .
            . . 3 .
            . . . 1
            4 . . .

        Returns:
            Array of shape (16,) with MASK at unknown positions.
        """
        x = np.full(16, MASK, dtype=np.int64)
        x[1] = 1   # (0,1) = 2 → value 1 (0-indexed)
        x[6] = 2   # (1,2) = 3 → value 2
        x[11] = 0  # (2,3) = 1 → value 0
        x[12] = 3  # (3,0) = 4 → value 3
        return x

    @staticmethod
    def puzzle_full() -> np.ndarray:
        """Return a known full valid 4×4 Sudoku grid for testing.

        Grid in standard notation:
            2 3 4 1
            4 1 2 3
            1 4 3 2
            3 2 1 4

        Returns:
            Array of shape (16,) with all positions filled.
        """
        x = np.array([
            1, 2, 3, 0,   # row 0: 2,3,4,1
            3, 0, 1, 2,   # row 1: 4,1,2,3
            0, 3, 2, 1,   # row 2: 1,4,3,2
            2, 1, 0, 3,   # row 3: 3,2,1,4
        ], dtype=np.int64)
        return x
