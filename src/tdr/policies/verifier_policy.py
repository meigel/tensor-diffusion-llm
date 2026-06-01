"""
Verifier-guided repair policy.

Remasks observed positions that participate in violated constraint groups,
identified by positive local residuals r_i(x) > 0.

This is the key policy for Milestone 4: verifier-guided remasking.
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import VerifierDiagnostics
from tdr.policies.entropy_policy import BaseMaskPolicy


class VerifierRepairPolicy(BaseMaskPolicy):
    """Remask positions with positive local residuals.

    After remasking, fills remaining MASK positions using denoiser
    confidence (same as ConfidenceUnmaskPolicy).

    Attributes:
        remask_threshold: Minimum local residual to trigger remasking.
                          Default 1 (any violation).
        top_k: If set, only remask the top-k highest-residual positions.
               If None, remask all positions with residual ≥ threshold.
    """

    def __init__(self, remask_threshold: int = 1, top_k: Optional[int] = None,
                 fill_threshold: float = 0.99):
        self.remask_threshold = remask_threshold
        self.top_k = top_k
        self.fill_threshold = fill_threshold

    def select_remask(self, x: np.ndarray,
                      diagnostics: VerifierDiagnostics,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Select observed positions with residual ≥ threshold.

        Only observed positions (x[i] != MASK) with positive local
        residuals are candidates for remasking.
        """
        n = len(x)
        residuals = diagnostics.local_residuals

        # Find observed positions with residual ≥ threshold
        candidates = np.where((x != MASK) & (residuals >= self.remask_threshold))[0]

        if len(candidates) == 0:
            return np.zeros(n, dtype=bool)

        if self.top_k is not None and len(candidates) > self.top_k:
            # Select top-k by residual
            indices = np.argsort(residuals[candidates])[::-1][:self.top_k]
            candidates = candidates[indices]

        remask = np.zeros(n, dtype=bool)
        remask[candidates] = True
        return remask

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Fill MASK positions with confidence ≥ threshold.

        Guarantees at least one fill per step.
        """
        n = len(x)
        masked = np.where(x == MASK)[0]
        if len(masked) == 0:
            return np.zeros(n, dtype=bool)

        confidences = dist[masked].max(axis=1)
        above_thresh = confidences >= self.fill_threshold

        fill = np.zeros(n, dtype=bool)
        fill[masked[above_thresh]] = True

        if not np.any(fill):
            best = masked[np.argmax(confidences)]
            fill[best] = True

        return fill


class RandomRemaskPolicy(BaseMaskPolicy):
    """Control baseline: remask a random subset of observed positions.

    Unlike VerifierRepairPolicy which uses local residuals to select
    positions for remasking, this policy remasks the same NUMBER of
    positions but chosen randomly. This isolates whether the
    verifier's LOCALIZATION matters or just the act of remasking.

    Attributes:
        remask_fraction: Fraction of observed positions to remask each step.
                         If None, remasks same fraction as verifier would
                         (not possible without knowing verifier output).
    """

    def __init__(self, remask_fraction: float = 0.25, fill_threshold: float = 0.99):
        self.remask_fraction = remask_fraction
        self.fill_threshold = fill_threshold

    def select_remask(self, x: np.ndarray,
                      diagnostics: VerifierDiagnostics,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Select a random subset of observed positions to remask."""
        if rng is None:
            rng = np.random.default_rng()
        n = len(x)
        observed = np.where(x != MASK)[0]
        if len(observed) == 0:
            return np.zeros(n, dtype=bool)
        k = max(1, int(len(observed) * self.remask_fraction))
        chosen = rng.choice(observed, size=k, replace=False)
        remask = np.zeros(n, dtype=bool)
        remask[chosen] = True
        return remask

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Fill MASK positions with confidence ≥ threshold."""
        n = len(x)
        masked = np.where(x == MASK)[0]
        if len(masked) == 0:
            return np.zeros(n, dtype=bool)
        confidences = dist[masked].max(axis=1)
        above_thresh = confidences >= self.fill_threshold
        fill = np.zeros(n, dtype=bool)
        fill[masked[above_thresh]] = True
        if not np.any(fill):
            best = masked[np.argmax(confidences)]
            fill[best] = True
        return fill


class ConfidenceRemaskPolicy(BaseMaskPolicy):
    """Baseline: remask positions where denoiser confidence is below a threshold.

    Unlike VerifierRepairPolicy which uses constraint residuals, this policy
    uses the denoiser's OWN uncertainty to decide what to remask.
    This is the standard approach in ReMDM-style remasking.
    """

    def __init__(self, remask_threshold: float = 0.5, fill_threshold: float = 0.99):
        self.remask_threshold = remask_threshold
        self.fill_threshold = fill_threshold

    def select_remask(self, x: np.ndarray,
                      diagnostics: VerifierDiagnostics,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """No pre-denoising remask (need denoiser output first)."""
        return np.zeros(len(x), dtype=bool)

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """After filling, also track low-confidence filled positions for next step."""
        return x == MASK  # Default: fill all masked positions, remask happens via separate pass


class ConfidenceFillThenRemask(BaseMaskPolicy):
    """Two-phase: fill high-confidence positions, then remask low-confidence ones.

    This matches the ReMDM-style approach: fill confident positions, then
    remask positions where confidence was low.
    """

    def __init__(self, fill_threshold: float = 0.99, remask_low_threshold: float = 0.5):
        self.fill_threshold = fill_threshold
        self.remask_low_threshold = remask_low_threshold

    def select_remask(self, x: np.ndarray,
                      diagnostics: VerifierDiagnostics,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """No remasking before denoising."""
        return np.zeros(len(x), dtype=bool)

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Fill all masked positions."""
        return x == MASK
