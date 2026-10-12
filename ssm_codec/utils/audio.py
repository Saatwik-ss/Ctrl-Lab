"""Audio signal processing utilities: mel spectrogram extraction and waveform synthesis."""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchaudio


class AudioProcessor:
    """Standardized 22.05 kHz audio processing and 80-bin log-mel extraction."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=1.0,
            normalized=False,
        )

        # Griffin-Lim inverse spectrogram transform for audio synthesis
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=32,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
        )

        # Approximate inverse mel filter bank for mel -> linear spectrogram inversion
        # Mel filter bank: (n_mels, n_fft // 2 + 1)
        mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )  # (n_freqs, n_mels)
        # Pseudoinverse: (n_mels, n_freqs) -> (n_freqs, n_mels)
        self.inv_mel_basis = torch.pinverse(mel_basis.t())  # (n_freqs, n_mels)

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Converts raw audio waveform to normalized log-mel spectrogram.

        Args:
            wav: (batch, samples) or (samples,) in [-1, 1]

        Returns:
            log_mel: (batch, n_mels, time)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Move mel transform to match wav device
        self.mel_transform = self.mel_transform.to(wav.device)
        mel = self.mel_transform(wav)
        # Dynamic range compression: log(clamp(mel, 1e-5))
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel

    def mel_to_wav(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Converts log-mel spectrogram to audio waveform via approximate inverse mel and Griffin-Lim.

        Args:
            log_mel: (batch, n_mels, time) or (n_mels, time)

        Returns:
            wav: (batch, samples)
        """
        if log_mel.dim() == 2:
            log_mel = log_mel.unsqueeze(0)

        device = log_mel.device
        mel = torch.exp(log_mel)  # (batch, n_mels, time)

        # Invert mel scale to linear STFT magnitude: (batch, n_freqs, time)
        inv_basis = self.inv_mel_basis.to(device)
        spec = torch.matmul(inv_basis, mel)
        spec = torch.clamp(spec, min=1e-5)

        self.griffin_lim = self.griffin_lim.to(device)
        wav = self.griffin_lim(spec)
        return wav


def generate_synthetic_speech(
    duration_sec: float = 2.0,
    sample_rate: int = 22050,
    speaker_f0: float = 140.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Generates synthetic speech-like acoustic signals with harmonic formant structure.

    Useful for offline testing, CI verification, and fast demonstration runs.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    t = torch.linspace(0, duration_sec, int(sample_rate * duration_sec))
    num_samples = len(t)

    # Glottal pulse train with fundamental frequency f0
    harmonics = [1.0, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05]
    signal = torch.zeros(num_samples)
    for i, amp in enumerate(harmonics, start=1):
        signal += amp * torch.sin(2.0 * math.pi * (speaker_f0 * i) * t)

    # Modulate formant envelope (simulating vowels: F1, F2 transitions)
    envelope = 0.5 + 0.5 * torch.sin(2.0 * math.pi * 3.0 * t)  # 3 Hz syllable rate
    signal = signal * envelope

    # Add subtle pink noise
    noise = torch.randn(num_samples) * 0.02
    signal = signal + noise

    # Normalize to [-0.95, 0.95]
    max_val = torch.max(torch.abs(signal))
    if max_val > 0:
        signal = 0.95 * signal / max_val

    return signal, sample_rate
