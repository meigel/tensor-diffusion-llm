"""
Mask policies for the diffusion sampler.

Each policy defines two methods:
  - select_remask(x, diagnostics, rng) → which observed positions to remask
  - select_fill(x, dist, diagnostics, rng) → which MASK positions to fill

Base class provides defaults (no remask, fill all MASK).
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import VerifierDiagnostics


class BaseMaskPolicy:
    """Base class for mask policies.

    Override select_remask for repair-mode policies.
    Override select_fill for custom denoising schedules.
    """

    def select_remask(self, x: np.ndarray,
                      diagnostics: VerifierDiagnostics,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return bool mask of observed positions to set to MASK (repair).

        Default: no remasking.
        """
        return np.zeros(len(x), dtype=bool)

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return bool mask of MASK positions to fill with argmax of dist.

        Default: fill all MASK positions.
        """
        return x == MASK


class ConfidenceUnmaskPolicy(BaseMaskPolicy):
    """Fill masked positions where denoiser confidence exceeds a threshold.

    Guarantees at least one fill per step (most confident position).
    No remasking.

    Attributes:
        threshold: Confidence threshold τ ∈ [0, 1].
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        n = len(x)
        masked = np.where(x == MASK)[0]
        if len(masked) == 0:
            return np.zeros(n, dtype=bool)

        confidences = dist[masked].max(axis=1)
        above_thresh = confidences >= self.threshold

        fill = np.zeros(n, dtype=bool)
        fill[masked[above_thresh]] = True

        # Guarantee at least one fill per step
        if not np.any(fill):
            best = masked[np.argmax(confidences)]
            fill[best] = True

        return fill


class RandomMaskPolicy(BaseMaskPolicy):
    """Randomly select a fraction of masked positions to fill.

    No remasking.

    Attributes:
        fraction: Fraction of masked positions to fill per step.
    """

    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        n = len(x)
        masked = np.where(x == MASK)[0]
        k = max(1, int(len(masked) * self.fraction))
        chosen = rng.choice(masked, size=k, replace=False)
        fill = np.zeros(n, dtype=bool)
        fill[chosen] = True
        return fill


class AllMaskPolicy(BaseMaskPolicy):
    """Fill all masked positions in a single step. No remasking."""

    def select_fill(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return x == MASK
