"""
Mask policies for the diffusion sampler.

Each policy defines a `select_mask` method that returns a boolean array
indicating which positions to unmask during the current step.
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import VerifierDiagnostics


class BaseMaskPolicy:
    """Base class for mask policies."""

    def select_mask(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return boolean mask of positions to unmask (or keep unmasked).

        Args:
            x: Current assignment.
            dist: Denoiser prediction distribution, shape (n, d).
            diagnostics: Verifier diagnostics.
            rng: Random number generator.

        Returns:
            Boolean array of shape (n,): True = keep/proceed with unmasking.
        """
        raise NotImplementedError


class ConfidenceUnmaskPolicy(BaseMaskPolicy):
    """Unmask positions where the denoiser confidence exceeds a threshold.

    Also forces unmask of the single most confident position each step
    to guarantee progress (c.f. plan Section 13.1).
    """

    def __init__(self, threshold: float = 0.99):
        self.threshold = threshold

    def select_mask(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        n = len(x)
        masked = np.where(x == MASK)[0]
        if len(masked) == 0:
            return np.zeros(n, dtype=bool)

        confidences = dist[masked].max(axis=1)
        above_thresh = confidences >= self.threshold

        unmask = np.zeros(n, dtype=bool)
        unmask[masked[above_thresh]] = True

        # Guarantee progress: if nothing above threshold, unmask the
        # single most confident masked position.
        if not np.any(unmask):
            best = masked[np.argmax(confidences)]
            unmask[best] = True

        return unmask


class RandomMaskPolicy(BaseMaskPolicy):
    """Randomly select a fraction of masked positions to unmask."""

    def __init__(self, fraction: float = 0.25):
        self.fraction = fraction

    def select_mask(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        n = len(x)
        masked = np.where(x == MASK)[0]
        k = max(1, int(len(masked) * self.fraction))
        chosen = rng.choice(masked, size=k, replace=False)
        unmask = np.zeros(n, dtype=bool)
        unmask[chosen] = True
        return unmask


class AllMaskPolicy(BaseMaskPolicy):
    """Unmask all masked positions in one step (direct filling)."""

    def select_mask(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return x == MASK
