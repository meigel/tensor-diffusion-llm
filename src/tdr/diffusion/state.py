"""
Diffusion state dataclass.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DiffusionState:
    """Mutable state for a masked diffusion run."""
    x: np.ndarray
    """Current assignment, shape (n,); entries in {0,...,d-1} or MASK."""
    step: int = 0
    """Current iteration step."""
    history: list = field(default_factory=list)
    """Log of per-step diagnostics."""
