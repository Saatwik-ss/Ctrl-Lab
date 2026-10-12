"""Test strict causality of the SSM Speech Codec layers.

Verifies that modifying future input frames (t > T_split) has zero effect
on outputs at earlier time frames (t <= T_split).
"""

import pytest
import torch
from ssm_codec.models.decoder import SSMDecoder
from ssm_codec.models.encoder import ContentEncoder


def test_encoder_causality():
    """Modifying future mel frames must not alter earlier encoded tokens."""
    torch.manual_seed(42)
    encoder = ContentEncoder(in_channels=80, model_dim=128, d_state=32, num_layers=4, latent_dim=64)
    encoder.eval()

    # Input mel with 64 frames (time divisible by 4)
    mel_a = torch.randn(1, 80, 64)
    mel_b = mel_a.clone()

    # Mutate only future frames: frames from index 32 onwards
    split_idx = 32
    mel_b[:, :, split_idx:] += torch.randn(1, 80, 64 - split_idx) * 5.0

    with torch.no_grad():
        z_a, _ = encoder(mel_a)
        z_b, _ = encoder(mel_b)

    # Downsampled time index corresponding to split_idx // 4 = 8
    # Because of downsampling filter kernel=4, past frames before split_idx // 4 must be unaffected
    downsampled_split = split_idx // 4
    # Check outputs up to downsampled_split - 1
    diff = torch.abs(z_a[:, :, :downsampled_split] - z_b[:, :, :downsampled_split]).max().item()
    assert diff < 1e-5, f"Encoder causality violated! Max diff on past frames: {diff}"


def test_decoder_causality():
    """Modifying future tokens must not alter earlier reconstructed mel frames."""
    torch.manual_seed(42)
    decoder = SSMDecoder(latent_dim=64, model_dim=128, d_state=32, num_layers=4, spk_dim=192, out_channels=80)
    decoder.eval()

    z_a = torch.randn(1, 64, 16)
    z_b = z_a.clone()
    spk_emb = torch.randn(1, 192)

    # Mutate future tokens from index 8 onwards
    split_idx = 8
    z_b[:, :, split_idx:] += torch.randn(1, 64, 16 - split_idx) * 5.0

    with torch.no_grad():
        mel_a, _ = decoder(z_a, spk_emb=spk_emb)
        mel_b, _ = decoder(z_b, spk_emb=spk_emb)

    # Upsampled time corresponds to split_idx * 4 = 32
    mel_split = split_idx * 4
    diff = torch.abs(mel_a[:, :, :mel_split] - mel_b[:, :, :mel_split]).max().item()
    assert diff < 1e-5, f"Decoder causality violated! Max diff on past mel frames: {diff}"
