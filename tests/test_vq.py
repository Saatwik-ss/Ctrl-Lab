"""Test Vector Quantizer functionality, straight-through gradients, and metrics."""

import pytest
import torch
from ssm_codec.modules.vq import VectorQuantizer


def test_vq_straight_through_and_loss():
    """Verifies gradient flow through STE and loss computation."""
    torch.manual_seed(42)
    vq = VectorQuantizer(num_embeddings=128, embedding_dim=64, commitment_cost=0.25)

    z_e = torch.randn(2, 64, 16, requires_grad=True)

    z_q, indices, metrics = vq(z_e)

    # Verify tensor shapes
    assert z_q.shape == (2, 64, 16)
    assert indices.shape == (2, 16)
    assert "vq_loss" in metrics
    assert "perplexity" in metrics

    # Verify gradients propagate to z_e
    loss = metrics["vq_loss"] + z_q.sum()
    loss.backward()

    assert z_e.grad is not None
    assert torch.sum(torch.abs(z_e.grad)) > 0.0


def test_decode_indices():
    """Verifies that decoding indices reproduces quantized representations."""
    vq = VectorQuantizer(num_embeddings=64, embedding_dim=32)
    indices = torch.tensor([[0, 5, 10], [1, 2, 3]])

    z_q = vq.decode_indices(indices)
    assert z_q.shape == (2, 32, 3)

    # Check that individual embedding matches codebook vector
    expected_vec = vq.embedding(torch.tensor(5))
    actual_vec = z_q[0, :, 1]
    assert torch.allclose(expected_vec, actual_vec, atol=1e-5)
