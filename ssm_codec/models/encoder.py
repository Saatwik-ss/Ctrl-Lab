"""Content Encoder with causal downsampling and stacked SSM layers."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_codec.modules.causal_ssm import CausalSSMLayer
from ssm_codec.modules.conv import CausalConv1d


class CausalConvDownsampler(nn.Module):
    """Causal convolutional downsampling front-end (factor 4x)."""

    def __init__(self, in_channels: int = 80, out_channels: int = 256):
        super().__init__()
        # Two cascaded stride-2 causal convs for stable 4x temporal downsampling
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
        )
        self.conv2 = CausalConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (batch, in_channels, time)

        Returns:
            downsampled: (batch, out_channels, time // 4)
        """
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return x


class ContentEncoder(nn.Module):
    """Content Encoder:

    Causal Conv (4x downsampling) -> 6 Causal SSM layers -> Linear projection to 64-d.
    """

    def __init__(
        self,
        in_channels: int = 80,
        model_dim: int = 256,
        d_state: int = 64,
        num_layers: int = 6,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_dim = model_dim
        self.d_state = d_state
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.downsampler = CausalConvDownsampler(
            in_channels=in_channels,
            out_channels=model_dim,
        )
        self.ssm_layers = CausalSSMLayer(
            num_layers=num_layers,
            d_model=model_dim,
            d_state=d_state,
        )
        # Linear projection to 64-d bottleneck
        self.proj = nn.Linear(model_dim, latent_dim)

    def init_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initializes recurrent states for streaming encoder."""
        return self.ssm_layers.init_states(batch_size, device)

    def forward(
        self,
        mel: torch.Tensor,
        ssm_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Args:

        mel: (batch, in_channels, time) mel spectrogram
        ssm_states: Optional list of recurrent states for streaming

        Returns:
            z_e: Continuous content latents (batch, latent_dim, time // 4)
            new_ssm_states: Updated states if streaming, else None
        """
        # Downsample by 4x
        h = self.downsampler(mel)  # (batch, model_dim, time // 4)

        # Process through 6 causal SSM layers
        h, new_states = self.ssm_layers(h, states=ssm_states)

        # Linear projection to 64-dim per frame
        # (batch, time // 4, model_dim) -> (batch, time // 4, latent_dim)
        z_e = self.proj(h.transpose(1, 2)).transpose(1, 2)

        return z_e, new_states
