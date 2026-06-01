"""
Tests for PoE (product-of-experts) and NoisyDenoiser.
"""

import numpy as np
import pytest

from tdr import MASK
from tdr.domains.sudoku4 import Sudoku4Domain
from tdr.tn.marginals import ContractionMarginalBackend
from tdr.diffusion.poe import product_of_experts, _safe_log
from tdr.diffusion.denoisers import (
    NoisyDenoiser,
    PoEDenoiser,
    TNMarginalDenoiser,
)

DOMAIN = Sudoku4Domain()
BACKEND = ContractionMarginalBackend(DOMAIN)


class TestProductOfExperts:
    """Tests for the PoE combination function."""

    def test_beta_1_equals_p(self):
        """beta=1 should return p unchanged."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet([1, 1, 1, 1], size=16)
        q = rng.dirichlet([1, 1, 1, 1], size=16)
        p_combined = product_of_experts(p, q, beta=1.0)
        assert np.allclose(p_combined, p), "beta=1 should give p"

    def test_beta_0_equals_q(self):
        """beta=0 should return q unchanged."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet([1, 1, 1, 1], size=16)
        q = rng.dirichlet([1, 1, 1, 1], size=16)
        p_combined = product_of_experts(p, q, beta=0.0)
        assert np.allclose(p_combined, q), "beta=0 should give q"

    def test_normalized(self):
        """Output should be normalized per position."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet([1, 1, 1, 1], size=16)
        q = rng.dirichlet([1, 1, 1, 1], size=16)
        for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p_combined = product_of_experts(p, q, beta=beta)
            sums = p_combined.sum(axis=-1)
            assert np.allclose(sums, 1.0), f"beta={beta}: sums={sums}"

    def test_beta_0_5_interpolates(self):
        """beta=0.5 should give intermediate result."""
        rng = np.random.default_rng(42)
        p = rng.dirichlet([10, 1, 1, 1], size=1)  # peaked
        q = rng.dirichlet([1, 10, 1, 1], size=1)  # peaked differently
        p_combined = product_of_experts(p, q, beta=0.5)
        assert not np.allclose(p_combined, p), "beta=0.5 should differ from p"
        assert not np.allclose(p_combined, q), "beta=0.5 should differ from q"

    def test_shape_mismatch_raises(self):
        """Different shapes should raise ValueError."""
        with pytest.raises(ValueError, match="shape"):
            product_of_experts(np.ones((16, 4)), np.ones((16, 3)), beta=0.5)

    def test_beta_out_of_range_raises(self):
        """beta outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="beta"):
            product_of_experts(
                np.ones((16, 4)), np.ones((16, 4)), beta=1.5
            )

    def test_delta_marginals_preserved(self):
        """Delta distributions should be approximately preserved.

        With p = [1, 0, 0, 0] (delta) and q uniform, PoE with beta=0.5
        should be very close to a delta (tiny residual from log(eps)).
        """
        p = np.zeros((16, 4))
        p[:, 0] = 1.0
        q = np.full((16, 4), 0.25)
        p_combined = product_of_experts(p, q, beta=0.5)
        assert np.allclose(p_combined[:, 0], 1.0, atol=1e-5)
        # Non-zero mass at other values should be negligible
        assert np.all(p_combined[:, 0] > 1.0 - 1e-5)

    def test_conflicting_zeros(self):
        """Conflicting hard zeros: each expert vetoes the other's choice.

        p = [1, 0] (delta on value 0), q = [0, 1] (delta on value 1).

        The product p(v)^β · q(v)^(1-β) is zero for both values when
        0 < β < 1, because each value is vetoed by one expert.
        At β=1 or β=0, the non-vetoing expert prevails.
        """
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.0, 1.0]])

        # beta=1: p only → [1, 0]
        assert np.allclose(product_of_experts(p, q, beta=1.0), p)

        # beta=0: q only → [0, 1]
        assert np.allclose(product_of_experts(p, q, beta=0.0), q)

        # beta=0.5: both experts veto each other's value → all zeros
        r = product_of_experts(p, q, beta=0.5)
        assert np.all(r == 0.0), f"Conflicting zeros should cancel: {r}"

        # beta=0.9: mostly p, but q still vetoes value 0 → all zeros
        r = product_of_experts(p, q, beta=0.9)
        assert np.all(r == 0.0), f"beta=0.9, q's zeros still block: {r}"

    def test_both_zeros_all_zero_row(self):
        """When both experts assign zero to all values, the row remains zero."""
        p = np.zeros((1, 4))
        q = np.zeros((1, 4))
        r = product_of_experts(p, q, beta=0.5)
        assert np.all(r == 0.0), f"All-zero row should stay zero: {r}"

    def test_safe_log_preserves_zeros(self):
        """_safe_log should map 0 → -inf and positive → finite."""
        x = np.array([0.0, 0.5, 1.0])
        lx = _safe_log(x)
        assert lx[0] == -np.inf
        assert np.isfinite(lx[1])
        assert np.isfinite(lx[2])

    def test_log_zero_multiplication(self):
        """-inf * 0 should be handled (beta=0 or beta=1 boundaries)."""
        p = np.array([[1.0, 0.0]])
        q = np.array([[0.3, 0.7]])

        # beta=0 uses only q, p's zeros don't matter
        r0 = product_of_experts(p, q, beta=0.0)
        assert np.allclose(r0, q)

        # beta=1 uses only p, q's zeros don't matter
        r1 = product_of_experts(p, q, beta=1.0)
        assert np.allclose(r1, p)


class TestNoisyDenoiser:
    """Tests for the NoisyDenoiser."""

    def test_sigma_0_equals_tn(self):
        """sigma=0 should reproduce exact TN marginals."""
        tn = TNMarginalDenoiser(BACKEND)
        noisy = NoisyDenoiser(BACKEND, sigma=0.0)
        x = DOMAIN.puzzle_easy()
        q_tn = tn.predict(x)
        q_noisy = noisy.predict(x)
        assert np.allclose(q_tn, q_noisy)

    def test_sigma_gt_0_differs(self):
        """sigma > 0 should produce different distributions from different noise draws."""
        noisy = NoisyDenoiser(BACKEND, sigma=2.0)
        # Use all-masked state so marginals are non-degenerate (uniform)
        x_masked = np.full(16, MASK, dtype=np.int64)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        q1 = noisy.predict(x_masked, rng=rng1)
        q2 = noisy.predict(x_masked, rng=rng2)
        # With sigma=2 and 16*4 logits, at least one should differ measurably
        max_diff = np.max(np.abs(q1 - q2))
        assert max_diff > 1e-3, f"Max diff too small: {max_diff}"

    def test_normalized(self):
        """Noisy distributions should be normalized."""
        noisy = NoisyDenoiser(BACKEND, sigma=0.5)
        x_masked = np.full(16, MASK, dtype=np.int64)
        q = noisy.predict(x_masked, rng=np.random.default_rng(0))
        assert np.allclose(q.sum(axis=-1), 1.0)

    def test_observed_delta_preserved(self):
        """Observed positions should remain delta distributions."""
        noisy = NoisyDenoiser(BACKEND, sigma=1.0)
        sol = DOMAIN.puzzle_full()
        q = noisy.predict(sol, rng=np.random.default_rng(0))
        for i in range(16):
            assert q[i, sol[i]] == 1.0
            assert np.isclose(q[i].sum(), 1.0)


class TestPoEDenoiser:
    """End-to-end tests for the PoEDenoiser."""

    def test_beta_1_equals_base(self):
        """PoEDenoiser with beta=1 should match the base denoiser."""
        base = NoisyDenoiser(BACKEND, sigma=0.5)
        poe = PoEDenoiser(base, BACKEND, beta=1.0)
        x = DOMAIN.puzzle_easy()
        rng = np.random.default_rng(42)
        p_base = base.predict(x, rng=rng)
        p_poe = poe.predict(x, rng=rng)
        assert np.allclose(p_base, p_poe)

    def test_beta_0_equals_tn(self):
        """PoEDenoiser with beta=0 should match TN marginal."""
        tn = TNMarginalDenoiser(BACKEND)
        base = NoisyDenoiser(BACKEND, sigma=0.5)  # base ignored at beta=0
        poe = PoEDenoiser(base, BACKEND, beta=0.0)
        x = DOMAIN.puzzle_easy()
        p_tn = tn.predict(x)
        p_poe = poe.predict(x)
        assert np.allclose(p_tn, p_poe)

    def test_completion(self):
        """PoEDenoiser should solve puzzles in the diffusion loop."""
        from tdr.diffusion.sampler import MaskedDiffusionSampler
        from tdr.policies.entropy_policy import ConfidenceUnmaskPolicy

        base = NoisyDenoiser(BACKEND, sigma=0.5)
        poe = PoEDenoiser(base, BACKEND, beta=0.5)
        sampler = MaskedDiffusionSampler(
            denoiser=poe,
            mask_policy=ConfidenceUnmaskPolicy(threshold=0.99),
            verifier=DOMAIN.verifier,
            max_steps=20,
        )
        successes = 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            sol = DOMAIN.sample_solution(rng)
            x_masked = DOMAIN.corrupt(sol, 0.5, rng)
            result = sampler.run(x_masked, rng)
            diag = DOMAIN.verifier(result.x)
            if diag.global_violation == 0 and np.all(result.x != MASK):
                successes += 1
        assert successes >= 8, f"PoE denoiser got {successes}/10 completions"
