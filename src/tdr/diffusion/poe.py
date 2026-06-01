"""
Product-of-experts combination for logic-guided denoising.

Combines a neural/noisy proposal distribution p_i(v) with a
tensor-network logical marginal q_i(v) via:

    log p̃_i(v) = β · log p_i(v) + (1-β) · log q_i(v) - log Z_i

where β ∈ [0,1] interpolates between pure neural denoising (β=1)
and pure logical inference (β=0).

Crucially, exact zeros in either distribution are preserved:
- If p_i(v) = 0, then log p_i(v) = -∞ and β · (-∞) = -∞ (for β > 0),
  so the combined value is 0 after softmax.
- If q_i(v) = 0 (logically impossible), the same applies.
- If all values are zero for a position (contradiction), the row
  remains zero (handled by the caller).
"""

import numpy as np


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Compute natural log, preserving zeros as -inf.

    log(0) = -inf is correct for the PoE combination: a zero
    probability means the expert rules out that value entirely,
    and it should remain ruled out in the product.
    """
    return np.log(x, out=np.full_like(x, -np.inf), where=(x > 0))


def product_of_experts(
    p: np.ndarray,
    q: np.ndarray,
    beta: float = 0.5,
) -> np.ndarray:
    """Combine two distributions by product-of-experts.

    Computes:

        log p̃_i(v) = β · log p_i(v) + (1-β) · log q_i(v)

    normalized over v for each position i.

    Exact zeros in either input are preserved as zeros in the output
    (via -inf in log space → 0 after softmax).

    Args:
        p:      First distribution array, shape (n, d).  The "neural/noisy"
                denoiser proposal.
        q:      Second distribution array, shape (n, d). The "logical"
                tensor-network marginal.
        beta:   Interpolation weight β ∈ [0, 1]. β=1 uses only p,
                β=0 uses only q.

    Returns:
        p_combined: Array of shape (n, d) of combined distributions,
                    normalized per position.

    Raises:
        ValueError: If p and q have different shapes or beta ∉ [0, 1].
    """
    if p.shape != q.shape:
        raise ValueError(
            f"p shape {p.shape} != q shape {q.shape}"
        )
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    # Log-space with exact zero preservation
    logp = _safe_log(p)
    logq = _safe_log(q)

    # Weighted combination in log space
    # -inf * β = -inf for β > 0; -inf * 0 = 0 with proper NaN handling
    if beta == 0.0:
        logits = logq
    elif beta == 1.0:
        logits = logp
    else:
        # Mask-based multiplication to handle -inf correctly
        logits = np.full_like(logp, -np.inf)
        finite_p = np.isfinite(logp)
        finite_q = np.isfinite(logq)
        both_finite = finite_p & finite_q
        logits[both_finite] = beta * logp[both_finite] + (1.0 - beta) * logq[both_finite]
        # If one expert is finite and the other -inf, the combined is -inf
        # (already set by initialisation)

    # Softmax normalisation (stable: subtract max before exp)
    row_max = logits.max(axis=-1, keepdims=True)
    # Avoid -inf - (-inf) = NaN for all-(-inf) rows
    has_finite = np.isfinite(row_max)
    logits_stable = np.where(has_finite, logits, 0.0)
    logits_stable = np.where(has_finite, logits_stable - row_max, logits_stable)
    out = np.exp(logits_stable)

    # Normalise (avoid division by zero for all-zero rows)
    row_sum = out.sum(axis=-1, keepdims=True)
    out = np.divide(out, row_sum, out=np.zeros_like(out), where=(row_sum > 0))

    # Rows where both experts assign zero to all values stay as all zeros
    # (not uniform, which would imply uncertainty rather than impossibility)
    out = np.where(has_finite, out, np.zeros_like(out))

    return out
