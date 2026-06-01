"""
Masked diffusion sampler: core iterative denoising loop.

The sampler follows the structure described in Section 5 of the
implementation plan. At each step k:

    1. Evaluate the current state through the verifier
    2. Compute denoiser proposal distribution
    3. Select positions to fill via the mask policy
    4. Fill selected positions with argmax of the proposal
    5. Record diagnostics and advance the step counter

The process terminates when either:
  - All positions are filled and no constraint violations remain, or
  - The maximum number of steps is reached.

Two modes are supported by choice of mask policy:
  - Mode 1 (Monotone Unmasking): Once filled, a position stays fixed.
  - Mode 2 (Full Remasking): The policy may re-mask any position.
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.diffusion.state import DiffusionState
from tdr.domains.base import FiniteReasoningDomain, VerifierDiagnostics


class MaskedDiffusionSampler:
    """Iterative masked diffusion loop.

    Implements the following algorithm at each step k:

        x^{(k)} → Verifier → diagnostics
                → Denoiser → proposal distribution p
                → MaskPolicy → unmask set U

        x^{(k+1)}_i = argmax p_i    if i ∈ U and x^{(k)}_i = MASK
                    = x^{(k)}_i     otherwise

    Attributes:
        denoiser:   Object with .predict(x_masked, rng) → (n, d) array.
        mask_policy: Object with .select_mask(x, dist, diagnostics) → bool mask.
        verifier:   Callable domain.verifier(x) → VerifierDiagnostics.
        max_steps:  Maximum refinement steps (early termination on success).
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
        """Run the masked diffusion loop from an initial state.

        Args:
            x_init: Initial assignment, shape (n,), with MASK at unknown positions.
            rng:    Random number generator (for stochastic policies/denoisers).

        Returns:
            DiffusionState with final assignment, step count, and per-step history.
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

            # Fill selected positions with argmax of the proposal
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
        """Compute verifier diagnostics for the current state."""
        return self.verifier(x)

    def _log_step(self, state: DiffusionState,
                  diagnostics: VerifierDiagnostics,
                  x_filled: np.ndarray,
                  num_masks: int) -> None:
        """Append a diagnostics record to the state history."""
        state.history.append({
            "step": state.step,
            "x": state.x.copy(),
            "num_masks": np.sum(state.x == MASK),
            "num_masks_after_fill": num_masks,
            "violation": diagnostics.global_violation,
        })
