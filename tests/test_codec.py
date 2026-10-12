"""Test end-to-end SSMSpeechCodec model, parameter count, and loss dynamics."""

import pytest
import torch
from ssm_codec.models.codec import SSMSpeechCodec


def test_parameter_count_budget():
    """Verifies that model parameter count aligns with the research proposal targets."""
    # 1. Compact configuration: targeted for strict single-GPU ~1M parameter constraint
    compact_model = SSMSpeechCodec(
        mel_bins=80,
        model_dim=128,
        d_state=32,
        num_layers=6,
        codebook_size=512,
        latent_dim=64,
        spk_dim=192,
        num_speakers=100,
    )
    compact_counts = compact_model.count_parameters()
    trainable_compact = compact_counts["trainable_total"]
    assert 1_000_000 <= trainable_compact <= 1_800_000, f"Unexpected compact parameter count: {trainable_compact:,}"

    # 2. Full configuration (Page 1 specs: model_dim=256, d_state=64, 6 layers)
    full_model = SSMSpeechCodec(
        mel_bins=80,
        model_dim=256,
        d_state=64,
        num_layers=6,
        codebook_size=512,
        latent_dim=64,
        spk_dim=192,
        num_speakers=100,
    )
    full_counts = full_model.count_parameters()
    trainable_full = full_counts["trainable_total"]
    assert 4_500_000 <= trainable_full <= 6_000_000, f"Unexpected full parameter count: {trainable_full:,}"


def test_codec_forward_and_losses():
    """Verifies forward pass and all four loss terms."""
    torch.manual_seed(42)
    model = SSMSpeechCodec(
        mel_bins=80,
        model_dim=128,
        d_state=32,
        num_layers=2,
        codebook_size=128,
        latent_dim=64,
        spk_dim=192,
        num_speakers=10,
    )

    batch_size = 2
    time_len = 32  # 32 mel frames
    mel = torch.randn(batch_size, 80, time_len)
    spk_ids = torch.tensor([0, 1])

    out = model(mel, spk_ids=spk_ids, grl_alpha=1.0)

    # Check outputs
    assert "mel_pred" in out
    assert out["mel_pred"].shape == (batch_size, 80, time_len)
    assert "total_loss" in out
    assert out["total_loss"].item() > 0.0

    # Check all four loss terms are present
    losses = out["losses"]
    assert "loss_rec" in losses
    assert "loss_vq" in losses
    assert "loss_leak" in losses
    assert "loss_consistency" in losses

    # Backward pass verification
    out["total_loss"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
