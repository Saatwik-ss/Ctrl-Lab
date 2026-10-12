"""Utility functions and helpers for SSM Speech Codec."""

from ssm_codec.utils.audio import AudioProcessor, generate_synthetic_speech
from ssm_codec.utils.metrics import (
    benchmark_rtf_and_latency,
    bootstrap_confidence_interval,
    compute_cosine_similarity,
)

__all__ = [
    "AudioProcessor",
    "benchmark_rtf_and_latency",
    "bootstrap_confidence_interval",
    "compute_cosine_similarity",
    "generate_synthetic_speech",
]
