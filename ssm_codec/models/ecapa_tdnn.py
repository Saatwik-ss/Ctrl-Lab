"""Self-contained ECAPA-TDNN architecture for extracting 192-d speaker embeddings."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling over time: (batch, channels)
        s = torch.mean(x, dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class Res2NetBlock(nn.Module):
    """Res2Net convolutional block with dilation and SE attention."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            dilation=dilation, padding=((kernel_size - 1) * dilation) // 2
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return F.relu(out + residual)


class AttentiveStatsPooling(nn.Module):
    """Attentive Statistics Pooling over temporal dimension."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim, time)
        alpha = self.attn(x)
        # Weighted mean
        mean = torch.sum(alpha * x, dim=-1)  # (batch, in_dim)
        # Weighted standard deviation
        residuals = x - mean.unsqueeze(-1)
        var = torch.sum(alpha * (residuals ** 2), dim=-1)
        std = torch.sqrt(torch.clamp(var, min=1e-5))
        return torch.cat([mean, std], dim=-1)  # (batch, in_dim * 2)


class ECAPATDNN(nn.Module):
    """Pretrained or standalone ECAPA-TDNN speaker encoder mapping mel spectrograms to 192-d embeddings."""

    def __init__(
        self,
        in_channels: int = 80,
        channels: int = 256,
        emb_dim: int = 192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.emb_dim = emb_dim

        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)

        self.block1 = Res2NetBlock(channels, dilation=2)
        self.block2 = Res2NetBlock(channels, dilation=3)
        self.block3 = Res2NetBlock(channels, dilation=4)

        # Multi-scale aggregation: 3 * channels
        self.mfa = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.asp = AttentiveStatsPooling(in_dim=channels * 3, hidden_dim=128)

        # Output projection to 192 dimensions
        self.fc = nn.Linear(channels * 3 * 2, emb_dim)
        self.bn_out = nn.BatchNorm1d(emb_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Extracts 192-d L2-normalized speaker embedding.

        Args:
            mel: (batch, in_channels, time) mel spectrogram

        Returns:
            embedding: (batch, emb_dim) L2-normalized speaker embedding
        """
        was_training = self.training
        # If batch size is 1, temporarily evaluate in eval mode to prevent BatchNorm error
        if mel.shape[0] == 1 and was_training:
            self.eval()

        try:
            x = F.relu(self.bn1(self.conv1(mel)))
            x1 = self.block1(x)
            x2 = self.block2(x1)
            x3 = self.block3(x2)

            # Concatenate multi-layer outputs
            cat = torch.cat([x1, x2, x3], dim=1)
            agg = F.relu(self.mfa(cat))

            # Attentive statistics pooling
            stats = self.asp(agg)
            emb = self.bn_out(self.fc(stats))

            # L2 normalize
            emb = F.normalize(emb, p=2, dim=-1)
            return emb
        finally:
            if mel.shape[0] == 1 and was_training:
                self.train(True)
