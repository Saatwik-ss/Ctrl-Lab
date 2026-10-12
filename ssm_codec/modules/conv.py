"""Strictly causal 1D convolution and transposed convolution modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Convolution with strict left-side causal padding.

    Ensures that for any time step t, output y[t] depends strictly on
    inputs x[tau] for tau <= t * stride. Never accesses future frames.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, out_channels, ceil(time / stride))
        """
        # Strict left-only causal padding (pad_left, pad_right)
        x_padded = F.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        return out


class CausalConvTranspose1d(nn.Module):
    """1D Transposed Convolution with causal alignment.

    Expands temporal resolution by factor `stride`. Excess receptive field
    is trimmed strictly from the right (future) so that no past sample
    is affected by future input tokens.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size >= stride, "kernel_size must be >= stride"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, out_channels, time * stride)
        """
        target_len = x.shape[-1] * self.stride
        out = self.deconv(x)
        # Trim from the right to maintain exact causal temporal alignment
        return out[..., :target_len]
