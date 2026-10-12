"""Decoder with FiLM speaker conditioning, SSM layers, and causal upsampling."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_codec.modules.causal_ssm import CausalSSMBlock
from ssm_codec.modules.conv import CausalConvTranspose1d
from ssm_codec.modules.film import FiLMBlock


class CausalConvUpsampler(nn.Module):
    """Causal transposed convolutional upsampling back-end (factor 4x)."""

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        # Two cascaded stride-2 causal conv transposes for 4x temporal upsampling
        self.deconv1 = CausalConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
        )
        self.deconv2 = CausalConvTranspose1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (batch, in_channels, time)

        Returns:
            upsampled: (batch, out_channels, time * 4)
        """
        x = F.silu(self.deconv1(x))
        x = F.silu(self.deconv2(x))
        return x


class FiLMSSMDecoderLayer(nn.Module):
    """Single SSM Decoder layer with FiLM speaker conditioning."""

    def __init__(self, model_dim: int = 256, d_state: int = 64, spk_dim: int = 192):
        super().__init__()
        self.film = FiLMBlock(spk_dim=spk_dim, feat_dim=model_dim)
        self.ssm_block = CausalSSMBlock(d_model=model_dim, d_state=d_state)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.ssm_block.init_state(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
        spk_emb: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Modulate feature representation with speaker identity via FiLM
        x = self.film(x, spk_emb)
        # Sequence modeling through causal SSM
        out, new_state = self.ssm_block(x, state=state)
        return out, new_state


class SSMDecoder(nn.Module):
    """Decoder mapping quantized tokens (z_q) and speaker embedding to 80-bin mel spectrogram."""

    def __init__(
        self,
        latent_dim: int = 64,
        model_dim: int = 256,
        d_state: int = 64,
        num_layers: int = 6,
        spk_dim: int = 192,
        out_channels: int = 80,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.spk_dim = spk_dim
        self.out_channels = out_channels

        # Project 64-dim bottleneck tokens up to model_dim
        self.in_proj = nn.Linear(latent_dim, model_dim)

        # 6 Causal SSM layers with FiLM conditioning per layer
        self.layers = nn.ModuleList([
            FiLMSSMDecoderLayer(model_dim=model_dim, d_state=d_state, spk_dim=spk_dim)
            for _ in range(num_layers)
        ])

        # Causal upsampling by 4x back to original mel frame rate
        self.upsampler = CausalConvUpsampler(in_channels=model_dim, out_channels=model_dim)

        # Output projection to 80-bin mel spectrogram
        self.out_proj = nn.Conv1d(model_dim, out_channels, kernel_size=1)

    def init_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initializes recurrent states for streaming decoder."""
        return [layer.init_state(batch_size, device) for layer in self.layers]

    def forward(
        self,
        z_q: torch.Tensor,
        spk_emb: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Args:

        z_q: (batch, latent_dim, time // 4) quantized tokens
        spk_emb: (batch, spk_dim) speaker embedding
        states: Optional list of recurrent states for each decoder layer

        Returns:
            mel_pred: (batch, out_channels, time) reconstructed mel spectrogram
            new_states: Updated states if streaming, else None
        """
        # Project tokens to model dimension
        h = self.in_proj(z_q.transpose(1, 2)).transpose(1, 2)  # (batch, model_dim, time // 4)

        new_states = [] if states is not None else None
        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            h, s = layer(h, spk_emb=spk_emb, state=layer_state)
            if new_states is not None:
                new_states.append(s)

        # Upsample temporally by 4x
        h_up = self.upsampler(h)  # (batch, model_dim, time)

        # Project to 80 mel channels
        mel_pred = self.out_proj(h_up)

        return mel_pred, new_states
