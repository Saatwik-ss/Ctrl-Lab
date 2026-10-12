"""Diagonal Causal State Space Model (S4D/LRU) in native PyTorch.

Supports two mathematically equivalent operational modes:
1. Parallel Mode (FFT convolution): O(L log L) forward pass during training.
2. Recurrent Mode (State stepping): O(1) state propagation for streaming inference.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSM(nn.Module):
    """Linear Time-Invariant Diagonal State-Space Model.

    Continuous formulation:
        dh/dt = A * h(t) + B * x(t)
        y(t)  = C * h(t) + D * x(t)

    Where A in C^{D x N} is diagonal with negative real parts to ensure stability.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize log_real(A) < 0 for Hurwitz stability
        # S4D-Lin initialization: real parts evenly spread, imaginary parts 0 to pi
        log_real = torch.log(0.5 * torch.ones(d_model, d_state))
        self.log_real = nn.Parameter(log_real)

        # Imaginary frequencies
        imag = math.pi * torch.arange(d_state).unsqueeze(0).repeat(d_model, 1)
        self.imag = nn.Parameter(imag)

        # Input matrix B: initialized standard normal complex
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / math.sqrt(d_state)))
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / math.sqrt(d_state)))

        # Output matrix C: initialized standard normal complex
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / math.sqrt(d_state)))
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) * (1.0 / math.sqrt(d_state)))

        # Feedthrough skip D
        self.d = nn.Parameter(torch.ones(d_model))

        # Discretization step Delta (dt) in log scale
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def get_discrete_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Discretizes continuous state space parameters (A, B, C) via Euler/ZOH.

        Returns:
            A_bar: (d_model, d_state) complex transition matrix
            B_bar: (d_model, d_state) complex input matrix
            C: (d_model, d_state) complex output matrix
        """
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (d_model, 1)

        # Enforce stability: Re(A) <= -exp(log_real)
        real_a = -torch.exp(self.log_real)
        a_complex = torch.complex(real_a, self.imag)

        # Discretization: A_bar = exp(dt * A)
        a_bar = torch.exp(dt * a_complex)

        # Input matrix B complex
        b_complex = torch.complex(self.b_real, self.b_imag)
        b_bar = b_complex * dt

        # Output matrix C complex
        c_complex = torch.complex(self.c_real, self.c_imag)

        return a_bar, b_bar, c_complex

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel FFT convolution mode.

        Args:
            x: Input tensor of shape (batch, d_model, time)

        Returns:
            y: Output tensor of shape (batch, d_model, time)
        """
        batch_size, d_model, length = x.shape
        a_bar, b_bar, c_complex = self.get_discrete_params()

        # Compute causal convolution kernel K of length L:
        # K_t = 2 * Re( sum_n (C_n * B_n * A_bar_n^t) )
        # Power computation: A_bar^t for t in 0..L-1
        t = torch.arange(length, device=x.device, dtype=torch.float32)  # (L,)

        # Log power for numerical stability: A_bar^t = exp(t * log(A_bar))
        log_a_bar = torch.log(a_bar)  # (d_model, d_state)
        # (L, d_model, d_state) = t[:, None, None] * log_a_bar[None, :, :]
        powers = torch.exp(t[:, None, None] * log_a_bar.unsqueeze(0))

        # CB product: (d_model, d_state)
        cb = c_complex * b_bar

        # Sum over state dimension: (L, d_model)
        kernel_complex = torch.sum(powers * cb.unsqueeze(0), dim=-1)
        kernel = 2.0 * kernel_complex.real  # (L, d_model)
        kernel = kernel.permute(1, 0)  # (d_model, L)

        # Perform causal convolution via FFT
        # Pad to 2*L for circular convolution avoidance
        fft_len = 2 * length
        x_fft = torch.fft.rfft(x, n=fft_len, dim=-1)
        k_fft = torch.fft.rfft(kernel, n=fft_len, dim=-1)  # (d_model, fft_bins)

        y_fft = x_fft * k_fft.unsqueeze(0)
        y = torch.fft.irfft(y_fft, n=fft_len, dim=-1)[..., :length]

        # Add feedthrough skip connection
        y = y + x * self.d.view(1, -1, 1)
        return y

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent step mode for streaming and chunked processing.

        Args:
            x: Chunk input tensor of shape (batch, d_model, chunk_len)
            state: Previous recurrent state (batch, d_model, d_state) complex, or None

        Returns:
            y: Chunk output tensor of shape (batch, d_model, chunk_len)
            new_state: Updated recurrent state (batch, d_model, d_state) complex
        """
        batch_size, d_model, chunk_len = x.shape
        a_bar, b_bar, c_complex = self.get_discrete_params()

        if state is None:
            state = torch.zeros(
                batch_size, d_model, self.d_state,
                dtype=torch.complex64, device=x.device
            )

        y_steps = []
        for t in range(chunk_len):
            x_t = x[:, :, t]  # (batch, d_model)
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            state = a_bar.unsqueeze(0) * state + b_bar.unsqueeze(0) * x_t.unsqueeze(-1).to(torch.complex64)
            # y_t = 2 * Re(C * h_t) + D * x_t
            y_t = 2.0 * torch.sum(c_complex.unsqueeze(0) * state, dim=-1).real + self.d.unsqueeze(0) * x_t
            y_steps.append(y_t)

        y = torch.stack(y_steps, dim=-1)  # (batch, d_model, chunk_len)
        return y, state

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes a zero recurrent state for streaming."""
        return torch.zeros(
            batch_size, self.d_model, self.d_state,
            dtype=torch.complex64, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unified forward pass.

        If state is provided, uses exact recurrent stepping and updates state.
        If state is None and training or offline inference, uses parallel FFT.
        """
        if state is not None:
            return self.forward_recurrent(x, state)
        else:
            return self.forward_parallel(x), None


class CausalSSMBlock(nn.Module):
    """Complete causal SSM block with LayerNorm, GELU/SiLU gating, and residual skip."""

    def __init__(self, d_model: int = 256, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.ssm = DiagonalSSM(d_model=d_model, d_state=d_state)
        self.out_proj = nn.Linear(d_model, d_model)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes recurrent state for streaming."""
        return self.ssm.init_state(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:

        x: (batch, d_model, time)
        state: Optional recurrent state
        """
        residual = x
        # Pre-LN: input expects (B, T, D)
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # Linear projection and gating
        # Project to (B, 2 * d_model, T)
        proj = self.in_proj(x_norm.transpose(1, 2)).transpose(1, 2)
        x_branch, gate = torch.chunk(proj, 2, dim=1)
        gate = F.silu(gate)

        # SSM processing
        y_ssm, new_state = self.ssm(x_branch, state=state)
        y = y_ssm * gate

        # Output projection and residual
        out = self.out_proj(y.transpose(1, 2)).transpose(1, 2)
        out = out + residual
        return out, new_state


class CausalSSMLayer(nn.Module):
    """Stack of N causal SSM blocks."""

    def __init__(self, num_layers: int = 6, d_model: int = 256, d_state: int = 64):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            CausalSSMBlock(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])

    def init_states(self, batch_size: int, device: torch.device) -> list:
        """Initializes recurrent states for all stacked SSM layers."""
        return [layer.init_state(batch_size, device) for layer in self.layers]

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass through all stacked SSM layers.

        Args:
            x: (batch, d_model, time)
            states: Optional list of recurrent states for each layer

        Returns:
            x: Output tensor (batch, d_model, time)
            new_states: List of updated states (or None)
        """
        new_states = [] if states is not None else None
        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            x, s = layer(x, state=layer_state)
            if new_states is not None:
                new_states.append(s)
        return x, new_states
