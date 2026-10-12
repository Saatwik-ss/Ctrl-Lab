"""Matched-parameter baseline decoders: Causal Transformer and GRU."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_codec.models.decoder import CausalConvUpsampler
from ssm_codec.modules.film import FiLMBlock


class CausalTransformerDecoderLayer(nn.Module):
    """Causal Transformer layer with FiLM speaker conditioning."""

    def __init__(self, model_dim: int = 256, nhead: int = 4, spk_dim: int = 192, dim_feedforward: int = 512):
        super().__init__()
        self.film = FiLMBlock(spk_dim=spk_dim, feat_dim=model_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, model_dim),
        )

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (batch, model_dim, time)
        spk_emb: (batch, spk_dim)
        """
        # Apply FiLM conditioning
        x = self.film(x, spk_emb)

        # Transpose to (batch, time, model_dim) for standard PyTorch attention
        x_t = x.transpose(1, 2)
        batch_size, time_len, _ = x_t.shape

        # Construct causal mask (True prevents attention to future positions)
        causal_mask = torch.triu(torch.ones(time_len, time_len, device=x.device), diagonal=1).bool()

        # Pre-LN Self-Attention
        norm_x = self.norm1(x_t)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=causal_mask, is_causal=True)
        x_t = x_t + attn_out

        # FFN
        x_t = x_t + self.ffn(self.norm2(x_t))
        return x_t.transpose(1, 2)


class TransformerDecoder(nn.Module):
    """Baseline A: Causal Transformer Decoder with parameter matching."""

    def __init__(
        self,
        latent_dim: int = 64,
        model_dim: int = 256,
        num_layers: int = 6,
        nhead: int = 4,
        spk_dim: int = 192,
        out_channels: int = 80,
    ):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.layers = nn.ModuleList([
            CausalTransformerDecoderLayer(model_dim=model_dim, nhead=nhead, spk_dim=spk_dim)
            for _ in range(num_layers)
        ])
        self.upsampler = CausalConvUpsampler(in_channels=model_dim, out_channels=model_dim)
        self.out_proj = nn.Conv1d(model_dim, out_channels, kernel_size=1)

    def forward(self, z_q: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z_q.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            h = layer(h, spk_emb=spk_emb)
        h_up = self.upsampler(h)
        return self.out_proj(h_up)


class CausalGRUDecoderLayer(nn.Module):
    """Causal GRU layer with FiLM speaker conditioning."""

    def __init__(self, model_dim: int = 256, spk_dim: int = 192):
        super().__init__()
        self.film = FiLMBlock(spk_dim=spk_dim, feat_dim=model_dim)
        self.gru = nn.GRU(input_size=model_dim, hidden_size=model_dim, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        x = self.film(x, spk_emb)
        x_t = x.transpose(1, 2)
        out_t, _ = self.gru(self.norm(x_t))
        out_t = out_t + x_t
        return out_t.transpose(1, 2)


class GRUDecoder(nn.Module):
    """Baseline B: Causal GRU Decoder with parameter matching."""

    def __init__(
        self,
        latent_dim: int = 64,
        model_dim: int = 256,
        num_layers: int = 6,
        spk_dim: int = 192,
        out_channels: int = 80,
    ):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, model_dim)
        self.layers = nn.ModuleList([
            CausalGRUDecoderLayer(model_dim=model_dim, spk_dim=spk_dim)
            for _ in range(num_layers)
        ])
        self.upsampler = CausalConvUpsampler(in_channels=model_dim, out_channels=model_dim)
        self.out_proj = nn.Conv1d(model_dim, out_channels, kernel_size=1)

    def forward(self, z_q: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z_q.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            h = layer(h, spk_emb=spk_emb)
        h_up = self.upsampler(h)
        return self.out_proj(h_up)
