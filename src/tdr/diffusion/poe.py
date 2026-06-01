"""
Product-of-experts combination for logic-guided denoising.

Combines a neural/noisy proposal distribution p_i(v) with a
tensor-network logical marginal q_i(v) via:

    log p̃_i(v) = β · log p_i(v) + (1-β) · log q_i(v) - log Z_i

where β ∈ [0,1] interpolates between pure neural denoising (β=1)
and pure logical inference (β=0).

For β=0.5, both sources contribute equally on a log-probability scale.
"""

import numpy as np


def product_of_experts(
    p: np.ndarray,
    q: np.ndarray,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Combine two distributions by product-of-experts.

    Computes:

        log p̃_i(v) = β · log p_i(v) + (1-β) · log q_i(v)

    normalized over v for each position i.

    Args:
        p:      First distribution array, shape (n, d).  The "neural/noisy"
                denoiser proposal.
        q:      Second distribution array, shape (n, d). The "logical"
                tensor-network marginal.
        beta:   Interpolation weight β ∈ [0, 1]. β=1 uses only p,
                β=0 uses only q.
        eps:    Small additive constant for numerical stability in log.

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

    # Compute log distributions with numerical stabiliser
    logp = np.log(p + eps)
    logq = np.log(q + eps)

    # Weighted combination in log space
    logits = beta * logp + (1.0 - beta) * logq

    # Softmax normalisation (stable: subtract max before exp)
    logits -= logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    out /= out.sum(axis=-1, keepdims=True)

    return out
