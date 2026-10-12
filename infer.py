"""Inference script for Zero-Shot Voice Conversion using SSM Speech Codec."""

import argparse
import os
import torch
import soundfile as sf

from ssm_codec.models.codec import SSMSpeechCodec
from ssm_codec.utils.audio import AudioProcessor, generate_synthetic_speech


def infer_conversion(
    source_wav_path: str = None,
    target_wav_path: str = None,
    output_wav_path: str = "output_converted.wav",
    device_str: str = "auto",
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    processor = AudioProcessor(sample_rate=22050, n_mels=80)

    # 1. Load or synthesize audio
    if source_wav_path and os.path.exists(source_wav_path):
        wav_src, sr_src = sf.read(source_wav_path)
        wav_src = torch.tensor(wav_src, dtype=torch.float32)
    else:
        print("[Infer] No source audio specified. Generating Speaker A (Male F0 ~ 120 Hz)...")
        wav_src, _ = generate_synthetic_speech(duration_sec=2.5, speaker_f0=120.0, seed=42)

    if target_wav_path and os.path.exists(target_wav_path):
        wav_tgt, sr_tgt = sf.read(target_wav_path)
        wav_tgt = torch.tensor(wav_tgt, dtype=torch.float32)
    else:
        print("[Infer] No target reference audio specified. Generating Speaker B (Female F0 ~ 230 Hz)...")
        wav_tgt, _ = generate_synthetic_speech(duration_sec=2.5, speaker_f0=230.0, seed=99)

    mel_src = processor.wav_to_mel(wav_src.to(device)).to(device)
    mel_tgt = processor.wav_to_mel(wav_tgt.to(device)).to(device)

    # Pad source to multiple of 4
    t_len = mel_src.shape[-1]
    pad = (4 - (t_len % 4)) % 4
    if pad > 0:
        mel_src = torch.nn.functional.pad(mel_src, (0, pad))

    # 2. Instantiate Codec
    model = SSMSpeechCodec(
        mel_bins=80,
        model_dim=128,
        d_state=32,
        num_layers=6,
        codebook_size=512,
        latent_dim=64,
        spk_dim=192,
    ).to(device)
    model.eval()

    # 3. Perform Disentangled Zero-Shot Voice Conversion
    with torch.no_grad():
        # Extract target speaker timbre
        spk_emb_tgt = model.extract_speaker_embedding(mel_tgt)

        # Encode linguistic tokens from source utterance
        z_q, indices, metrics, _ = model.encode(mel_src)

        # Synthesize mel with target speaker FiLM conditioning
        mel_pred, _ = model.decode(z_q, spk_emb=spk_emb_tgt)

        # Synthesize audio waveform
        wav_converted = processor.mel_to_wav(mel_pred.squeeze(0).cpu())

    # 4. Save converted audio
    sf.write(output_wav_path, wav_converted.squeeze().numpy(), 22050)
    print(f"[Infer] Converted audio successfully generated and saved to: {output_wav_path}")
    print(f"  Source duration: {len(wav_src)/22050:.2f}s | Output duration: {len(wav_converted.squeeze())/22050:.2f}s")
    print(f"  Content tokens bitrate: ~194 bps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Voice Conversion Inference")
    parser.add_argument("--source", type=str, default=None, help="Path to source content wav")
    parser.add_argument("--target", type=str, default=None, help="Path to target reference speaker wav")
    parser.add_argument("--output", type=str, default="converted_output.wav", help="Output wav file")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    infer_conversion(args.source, args.target, args.output, args.device)
