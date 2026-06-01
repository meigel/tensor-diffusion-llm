"""
Masked diffusion sampler: core iterative denoising loop.

The sampler supports two phases per step:
  1. REMASK: the policy can select observed positions to return to MASK
     (repair mode, based on verifier diagnostics).
  2. FILL: denoise on the resulting state, then fill selected MASK
     positions with the argmax of the proposal distribution.

Policies implement two methods:
  - select_remask(x, diagnostics, rng) → positions to remask (default: none)
  - select_fill(x, dist, diagnostics, rng) → positions to fill (default: all MASK)
"""

from typing import Optional

import numpy as np

from tdr import MASK
from tdr.diffusion.state import DiffusionState
from tdr.domains.base import FiniteReasoningDomain, VerifierDiagnostics


class MaskedDiffusionSampler:
    """Iterative masked diffusion loop with remasking support.

    At each step k:
      1. Verifier diagnostics on current state.
      2. Policy selects positions to REMASK (observed → MASK).
      3. Denoiser prediction on the resulting state.
      4. Policy selects positions to FILL (MASK → argmax of proposal).
      5. Record diagnostics and advance.
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
        """Run the diffusion loop from an initial state.

        Args:
            x_init: Initial assignment with MASK at unknown positions.
            rng:    Random number generator.

        Returns:
            DiffusionState with final assignment and history.
        """
        if rng is None:
            rng = np.random.default_rng()

        state = DiffusionState(x=x_init.copy(), step=0, history=[])

        for k in range(self.max_steps):
            diagnostics = self._compute_diagnostics(state.x)

            # Check termination
            if np.all(state.x != MASK) and diagnostics.global_violation == 0:
                self._log_step(state, diagnostics)
                break

            # Phase 1: remask violated observed positions
            remask = self.mask_policy.select_remask(
                state.x, diagnostics, rng=rng,
            )
            x_after_remask = state.x.copy()
            x_after_remask[remask] = MASK

            # Phase 2: denoise on the (possibly re-masked) state
            dist = self.denoiser.predict(x_after_remask, rng=rng)

            # Phase 3: select positions to fill
            fill = self.mask_policy.select_fill(
                x_after_remask, dist, diagnostics, rng=rng,
            )

            # Fill selected MASK positions with argmax
            x_filled = x_after_remask.copy()
            for i in range(len(state.x)):
                if fill[i] and x_after_remask[i] == MASK:
                    x_filled[i] = int(np.argmax(dist[i]))

            # Record step
            self._log_step(state, diagnostics, x_filled,
                           int(np.sum(x_filled == MASK)),
                           int(np.sum(remask)))

            state.x = x_filled
            state.step += 1

        return state

    def _compute_diagnostics(self, x: np.ndarray) -> VerifierDiagnostics:
        return self.verifier(x)

    def _log_step(self, state: DiffusionState,
                  diagnostics: VerifierDiagnostics,
                  x_filled: Optional[np.ndarray] = None,
                  num_masks_after_fill: Optional[int] = None,
                  num_remasked: Optional[int] = None) -> None:
        record = {
            "step": state.step,
            "x": state.x.copy(),
            "num_masks": int(np.sum(state.x == MASK)),
            "violation": diagnostics.global_violation,
        }
        if x_filled is not None:
            record["x_filled"] = x_filled.copy()
            record["num_masks_after_fill"] = num_masks_after_fill
            record["num_remasked"] = num_remasked
        state.history.append(record)
