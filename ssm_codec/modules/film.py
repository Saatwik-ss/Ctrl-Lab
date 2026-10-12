"""Feature-wise Linear Modulation (FiLM) layer for speaker conditioning."""

import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Applies affine transformation conditioned on speaker embedding:
        y = (1 + gamma) * x + beta
    where gamma and beta are learned projections of the speaker embedding.
    """

    def __init__(self, spk_dim: int = 192, feat_dim: int = 256):
        super().__init__()
        self.spk_dim = spk_dim
        self.feat_dim = feat_dim

        # Project 192-dim speaker embedding to (gamma, beta)
        self.proj = nn.Linear(spk_dim, feat_dim * 2)

        # Initialize to identity transform (gamma=0, beta=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """Modulates feature representation x with speaker embedding spk_emb.

        Args:
            x: Hidden representations of shape (batch, feat_dim, time)
            spk_emb: Speaker embedding of shape (batch, spk_dim)

        Returns:
            Modulated tensor of shape (batch, feat_dim, time)
        """
        # (batch, feat_dim * 2)
        params = self.proj(spk_emb)
        gamma, beta = torch.chunk(params, 2, dim=-1)

        # Reshape for broadcasting over time dimension (batch, feat_dim, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Affine conditioning with identity base (1 + gamma)
        return (1.0 + gamma) * x + beta
