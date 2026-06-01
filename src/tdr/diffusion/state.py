"""
Diffusion state dataclass.

Tracks the evolving masked state x^{(k)} through the iterative
denoising process:

    x^{(k)} ∈ (A ∪ {MASK})ⁿ
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DiffusionState:
    """Mutable state for a masked diffusion run.

    The state vector x evolves from a partially masked initialisation
    toward a valid full assignment through successive denoising steps.

    Attributes:
        x:       Current assignment, shape (n,).
                 Entries in {0, ..., d-1} (observed) or MASK=-1 (masked).
        step:    Current iteration index k (0-indexed).
        history: Log of per-step diagnostics (verifier output, mask counts).
    """
    x: np.ndarray
    """Current assignment vector x^{(k)} of shape (n,)."""
    step: int = 0
    """Current iteration step k of the diffusion process."""
    history: list = field(default_factory=list)
    """Per-step diagnostics log.

    Each entry is a dict with keys:
        step, x, num_masks, num_masks_after_fill, violation
    """
