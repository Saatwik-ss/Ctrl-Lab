"""Test numerical equivalence of offline batch processing vs chunked recurrent streaming."""

import pytest
import torch
from ssm_codec.modules.causal_ssm import DiagonalSSM, CausalSSMBlock


def test_ssm_streaming_equivalence():
    """Verifies that chunked recurrent inference produces output equivalent to offline FFT."""
    torch.manual_seed(42)
    d_model = 64
    d_state = 32
    length = 64
    chunk_size = 8

    ssm = DiagonalSSM(d_model=d_model, d_state=d_state)
    ssm.eval()

    x = torch.randn(2, d_model, length)

    with torch.no_grad():
        # 1. Full offline forward pass (parallel FFT)
        y_offline, _ = ssm(x, state=None)

        # 2. Sequential chunked recurrent pass
        state = ssm.init_state(x.shape[0], x.device)
        y_chunks = []
        for t in range(0, length, chunk_size):
            chunk = x[:, :, t : t + chunk_size]
            y_c, state = ssm(chunk, state=state)
            y_chunks.append(y_c)
        y_stream = torch.cat(y_chunks, dim=-1)

    max_err = torch.abs(y_offline - y_stream).max().item()
    assert max_err < 1e-4, f"Streaming numerical mismatch! Max error: {max_err}"


def test_ssm_block_streaming():
    """Verifies CausalSSMBlock streaming with recurrent state propagation."""
    torch.manual_seed(42)
    d_model = 64
    d_state = 32
    length = 32
    chunk_size = 4

    block = CausalSSMBlock(d_model=d_model, d_state=d_state)
    block.eval()

    x = torch.randn(1, d_model, length)

    with torch.no_grad():
        y_offline, _ = block(x, state=None)

        state = block.init_state(x.shape[0], x.device)
        y_chunks = []
        for t in range(0, length, chunk_size):
            chunk = x[:, :, t : t + chunk_size]
            y_c, state = block(chunk, state=state)
            y_chunks.append(y_c)
        y_stream = torch.cat(y_chunks, dim=-1)

    max_err = torch.abs(y_offline - y_stream).max().item()
    assert max_err < 1e-4, f"SSM Block streaming mismatch! Max error: {max_err}"
