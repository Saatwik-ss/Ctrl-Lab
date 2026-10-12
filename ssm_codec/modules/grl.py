"""Gradient Reversal Layer (GRL) and Speaker Classifier for adversarial disentanglement."""

from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Autograd function reversing gradients during backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse and scale gradient
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module applying Gradient Reversal Function."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        scale = alpha if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, scale)


class SpeakerClassifier(nn.Module):
    """Adversarial speaker classifier operating on quantized content latents.

    Trained to predict speaker ID from content tokens. With GRL placed before it,
    the encoder is pushed to minimize speaker-predictive information in content tokens.
    """

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        num_speakers: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_speakers),
        )

    def forward(
        self,
        z_q: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Args:

        z_q: Quantized content latents of shape (batch, in_dim, time)
        alpha: Gradient reversal scaling parameter

        Returns:
            logits: Speaker classification logits of shape (batch, num_speakers)
        """
        # Pool across time to aggregate utterance-level representation
        # (batch, in_dim)
        z_pooled = torch.mean(z_q, dim=-1)

        # Apply gradient reversal to invert encoder gradients
        z_reversed = self.grl(z_pooled, alpha=alpha)

        # Classify speaker
        logits = self.classifier(z_reversed)
        return logits
