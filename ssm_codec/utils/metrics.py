"""Evaluation metrics, RTF benchmarking, and bootstrap confidence intervals."""

import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Computes mean cosine similarity between speaker embeddings."""
    cos = F.cosine_similarity(emb1, emb2, dim=-1)
    return float(cos.mean().item())


def benchmark_rtf_and_latency(
    model: torch.nn.Module,
    mel: torch.Tensor,
    spk_emb: torch.Tensor,
    warmup: int = 5,
    runs: int = 20,
    hop_length: int = 256,
    sample_rate: int = 22050,
) -> Dict[str, float]:
    """Measures Real-Time Factor (RTF) and per-frame latency.

    RTF = total_compute_time / audio_duration_seconds
    RTF < 1.0 indicates faster than real time.
    """
    model.eval()
    device = next(model.parameters()).device
    mel = mel.to(device)
    spk_emb = spk_emb.to(device)

    # Calculate audio duration
    num_frames = mel.shape[-1]
    audio_duration_sec = (num_frames * hop_length) / sample_rate

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            z_q, _, _, _ = model.encode(mel)
            _ = model.decode(z_q, spk_emb=spk_emb)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed benchmark
        start_time = time.perf_counter()
        for _ in range(runs):
            z_q, _, _, _ = model.encode(mel)
            _ = model.decode(z_q, spk_emb=spk_emb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()

    avg_time = (end_time - start_time) / runs
    rtf = avg_time / audio_duration_sec
    per_frame_ms = (avg_time / num_frames) * 1000.0

    return {
        "audio_duration_sec": audio_duration_sec,
        "avg_compute_sec": avg_time,
        "rtf": rtf,
        "per_frame_latency_ms": per_frame_ms,
    }


def bootstrap_confidence_interval(
    data: List[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes sample mean and 95% bootstrap confidence intervals [ci_lower, ci_upper]."""
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    mean_val = float(np.mean(arr))
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_val, ci_lower, ci_upper
