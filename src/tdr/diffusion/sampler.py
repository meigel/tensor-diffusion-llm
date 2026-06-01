"""
Masked diffusion sampler: core loop that iteratively denoises and unmasks.

Follows the design in Section 5 of the implementation plan.
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.diffusion.state import DiffusionState
from tdr.domains.base import FiniteReasoningDomain, VerifierDiagnostics


class MaskedDiffusionSampler:
    """Iterative masked diffusion loop.

    At each step:
      1. Run the denoiser on the current masked state.
      2. Select positions to unmask via the mask policy.
      3. Fill selected positions with argmax of the denoiser distribution.
      4. Record diagnostics.
    """

    def __init__(
        self,
        denoiser,
        mask_policy,
        verifier,
        max_steps: int = 20,
    ):
        self.denoiser = denoiser
        self.mask_policy = mask_policy
        self.verifier = verifier
        self.max_steps = max_steps

    def run(self, x_init: np.ndarray, rng: Optional[np.random.Generator] = None) -> DiffusionState:
        """Run the diffusion loop from an initial masked state.

        Args:
            x_init: Initial assignment with some positions set to MASK.
            rng: Random number generator.

        Returns:
            DiffusionState with final assignment and history.
        """
        if rng is None:
            rng = np.random.default_rng()

        state = DiffusionState(x=x_init.copy(), step=0, history=[])

        for k in range(self.max_steps):
            diagnostics = self._compute_diagnostics(state.x)

            # Check termination: fully assigned and no violations
            if np.all(state.x != MASK) and diagnostics.global_violation == 0:
                self._log_step(state, diagnostics, state.x, np.sum(state.x == MASK))
                break

            # Denoiser prediction
            dist = self.denoiser.predict(state.x, rng=rng)

            # Select positions to unmask
            unmask = self.mask_policy.select_mask(
                state.x, dist, diagnostics, rng=rng,
            )

            # Fill selected positions with argmax
            n = len(state.x)
            x_filled = state.x.copy()
            for i in range(n):
                if unmask[i] and state.x[i] == MASK:
                    x_filled[i] = int(np.argmax(dist[i]))

            # Record step
            self._log_step(state, diagnostics, x_filled, np.sum(x_filled == MASK))

            state.x = x_filled
            state.step += 1

        return state

    def _compute_diagnostics(self, x: np.ndarray) -> VerifierDiagnostics:
        return self.verifier(x)

    def _log_step(self, state: DiffusionState,
                  diagnostics: VerifierDiagnostics,
                  x_filled: np.ndarray,
                  num_masks: int) -> None:
        # Estimate entropy over masked positions
        state.history.append({
            "step": state.step,
            "x": state.x.copy(),
            "num_masks": np.sum(state.x == MASK),
            "num_masks_after_fill": num_masks,
            "violation": diagnostics.global_violation,
        })
