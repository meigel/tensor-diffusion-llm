"""
Minimal transformer MDLM denoiser for masked diffusion.

Provides a TransformerDenoiserModel (PyTorch nn.Module) and a
MDLMTransformerDenoiser wrapper with the predict() interface
compatible with the diffusion sampler.

Architecture
------------
    Input:  one-hot encoded state, shape (n × (d+1),)
            where d is max domain size and the extra channel encodes MASK.

    The flat vector is reshaped to (n, d+1), projected to an embedding,
    augmented with learned positional encodings, passed through
    bidirectional transformer encoder layers, and projected to
    output logits over d values per position.

    This is a genuine MDLM (Masked Diffusion Language Model) following
    the D3PM / MDLM paradigm: a bidirectional masked-prediction model
    trained via masked language modelling on discrete sequences.

Rationale
---------
Unlike the MLP denoiser (which processes all variables as a flat vector
without positional structure), the transformer explicitly models
inter-variable interactions through self-attention. For JSON sequences
with cross-field constraints (e.g., admin requires high clearance),
this should learn the field relationships from data.

References
----------
- Austin et al., "Structured Denoising Diffusion Models in Discrete
  State-Spaces" (D3PM), NeurIPS 2021.
- Shi et al., "MDLM: Masked Diffusion Language Model", 2024.
"""

import numpy as np
import torch
import torch.nn as nn
from tdr import MASK


class TransformerDenoiserModel(nn.Module):
    """Minimal bidirectional transformer for masked denoising.

    Encodes a masked sequence and predicts logits over the domain
    for each position. Follows the MDLM (Masked Diffusion Language
    Model) architecture: bidirectional self-attention, no causal mask.

    Parameters
    ----------
    n : int
        Number of variables (sequence length).
    d : int
        Maximum domain size (vocabulary size per position).
    embed_dim : int
        Embedding dimension (default 128).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of transformer encoder layers (default 3).
    dim_feedforward : int
        Hidden dimension of the FFN sub-layer (default 512).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        n: int,
        d: int,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.d = d
        self.embed_dim = embed_dim

        # Input projection: one-hot (d+1 channels) -> embedding
        self.input_proj = nn.Linear(d + 1, embed_dim)

        # Learned positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, n, embed_dim) * 0.02
        )

        # Bidirectional transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head: embedding -> logits over d values
        self.output_head = nn.Linear(embed_dim, d)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with N(0, 0.02) following GPT-style init."""
        for module in [self.input_proj, self.output_head]:
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Flat one-hot state, shape (batch, n × (d+1)).

        Returns:
            logits: Shape (batch, n × d), logits over d values
                    for each position, concatenated.
        """
        batch_size = x.shape[0]

        # Reshape to sequence: (batch, n, d+1)
        x = x.reshape(batch_size, self.n, self.d + 1)

        # Project to embedding space
        x = self.input_proj(x)  # (batch, n, embed_dim)

        # Add learned positional encoding
        x = x + self.pos_embedding

        # Bidirectional transformer encoder
        x = self.transformer(x)  # (batch, n, embed_dim)

        # Output logits per position
        logits = self.output_head(x)  # (batch, n, d)

        # Flatten to (batch, n*d) for compatibility with training loop
        return logits.reshape(batch_size, self.n * self.d)


class MDLMTransformerDenoiser:
    """Wrapper around TransformerDenoiserModel for the denoising sampler.

    Converts between the sampler's integer-coded state (n,) with MASK
    and the transformer's one-hot encoding, runs inference, and returns
    (n, d) probability distributions.

    For domains with heterogeneous per-position domain sizes (e.g. JSON
    where age has 63 values but role has 4), invalid logits are masked
    to -inf before softmax so the predictor never emits impossible values.

    Args:
        model: Trained TransformerDenoiserModel.
        n: Number of variables.
        d: Maximum domain size (used for array shapes).
        domain_sizes: Optional list of per-variable domain sizes.
                      If provided, logits for each position i are
                      restricted to the first domain_sizes[i] values.
                      If None (default), all d values are used.
    """

    def __init__(
        self,
        model: TransformerDenoiserModel,
        n: int,
        d: int,
        domain_sizes: list[int] | None = None,
    ):
        self.model = model
        self.model.eval()
        self.n = n
        self.d = d
        self.domain_sizes = domain_sizes
        self._device = next(model.parameters()).device

    def predict(
        self,
        x_masked: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return denoiser predictions for the masked state.

        For heterogeneous domains, invalid classes (beyond each
        position's domain_size) are masked to -inf before softmax,
        so the sampler never receives impossible proposals.

        Args:
            x_masked: State array, shape (n,); entries in
                      {0, ..., d-1} or MASK (-1).
            rng: Ignored (included for interface compatibility).

        Returns:
            q: Array of shape (n, d) of predicted probabilities.
               Observed positions have a delta distribution at their
               known value.
        """
        n, d = self.n, self.d

        # Build one-hot encoding: d value channels + 1 MASK channel
        one_hot = torch.zeros(
            1, n * (d + 1), dtype=torch.float32, device=self._device
        )
        for i in range(n):
            if x_masked[i] == MASK:
                one_hot[0, i * (d + 1) + d] = 1.0  # MASK channel
            else:
                one_hot[0, i * (d + 1) + x_masked[i]] = 1.0  # value channel

        with torch.no_grad():
            logits = self.model(one_hot)  # (1, n*d)

        logits = logits.reshape(1, n, d)

        # Domain mask: for heterogeneous domains, suppress invalid classes
        if self.domain_sizes is not None:
            for i in range(n):
                if self.domain_sizes[i] < d:
                    logits[0, i, self.domain_sizes[i]:] = float("-inf")

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Override observed positions with delta distributions
        for i in range(n):
            if x_masked[i] != MASK:
                probs[i, :] = 0.0
                probs[i, x_masked[i]] = 1.0

        return probs
