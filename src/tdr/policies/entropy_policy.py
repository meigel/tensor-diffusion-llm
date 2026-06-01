"""
Mask policies for the diffusion sampler.

Each policy defines how positions are selected for unmasking (or remasking)
at a given diffusion step.

A mask policy receives:
  - The current state x
  - The denoiser proposal distribution p_i(v) [shape (n, d)]
  - Verifier diagnostics (violations, residuals)
  - An optional RNG

and returns a boolean array indicating which positions to act on.

Policy types
------------
1. ConfidenceUnmaskPolicy — unmask positions with max confidence ≥ threshold
2. RandomMaskPolicy — select a random fraction of masked positions
3. AllMaskPolicy — unmask all masked positions in one step
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.domains.base import VerifierDiagnostics


class BaseMaskPolicy:
    """Base class for mask policies.

    A policy's select_mask method determines which positions to unmask
    (or keep unmasked) at each diffusion step.
    """

    def select_mask(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return boolean array indicating which positions to unmask.

        Args:
            x: Current assignment vector, shape (n,).
            dist: Denoiser proposal distribution, shape (n, d).
            diagnostics: Verifier diagnostics (violations, residuals).
            rng: Random number generator for stochastic policies.

        Returns:
            Boolean array of shape (n,): True means "unmask this position"
            (or keep it unmasked if already observed).
        """
        raise NotImplementedError


class ConfidenceUnmaskPolicy(BaseMaskPolicy):
    """Unmask positions where the denoiser confidence exceeds a threshold.

    Let c_i = max_v p_i(v) be the confidence for masked position i.
    Positions with c_i ≥ τ are unmasked.

    If no positions meet the threshold, the single most confident masked
    position is unmasked to guarantee progress (c.f. plan Section 13.1).

    Attributes:
        threshold: Confidence threshold τ ∈ [0, 1].
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

        # Guarantee at least one unmask per step
        if not np.any(unmask):
            best = masked[np.argmax(confidences)]
            unmask[best] = True

        return unmask


class RandomMaskPolicy(BaseMaskPolicy):
    """Randomly select a fraction of masked positions to unmask.

    Selects ceil(α · |M|) positions uniformly at random, where
    M is the set of currently masked positions and α is the fraction.

    Attributes:
        fraction: Fraction α of masked positions to unmask per step.
    """

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
    """Unmask all masked positions in a single step (direct filling).

    Equivalent to one-shot generation: denoise → fill all in one step.
    Useful for establishing an upper bound on single-step performance.
    """

    def select_mask(self, x: np.ndarray, dist: np.ndarray,
                    diagnostics: VerifierDiagnostics,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return x == MASK
