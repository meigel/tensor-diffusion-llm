"""
Tensor-guided masked diffusion for logical sequence reasoning.

Package: tdr (tensor-diffusion-reasoning)

This framework combines masked diffusion-style denoising with logical
tensor-network (TN) reasoning and adaptive mask policies to iteratively
repair partially masked symbolic sequences.

Mathematical setting
--------------------
Let x ∈ (A ∪ {MASK})ⁿ be a masked state over finite domain A of size d,
with MASK = -1 denoting unobserved positions. The state evolves through:

    x^{(k+1)} = MaskPolicy · Denoise · Verifier (x^{(k)})

where at each step:
  1. A denoiser produces a proposal distribution p_i(v) per masked position
  2. A mask policy selects positions to fill (or remask)
  3. A verifier computes constraint violations to guide the process

The central mechanism is the product-of-experts correction:

    log p̃_i(v) = β·log p_i(v) + (1-β)·log q_i(v) - log Z_i

combining neural/noisy proposals p_i with exact TN marginals q_i.
"""

MASK = -1
"""Sentinel value indicating a masked (unobserved) variable position.

The state vector uses this sentinel convention:
    x_i ∈ {-1} ∪ {0, 1, ..., d-1}
where -1 = MASK means unobserved and 0..d-1 are valid domain values.
"""
