"""Professional Research Demonstration: SSM Speech Codec with Zero-Shot Speaker Conditioning.

Project: IISc SPIRE Lab Research Application (Speech AI)
Demonstrates:
  1. Full model instantiation & component parameter audit (~1M parameter budget).
  2. Audio feature extraction (80-bin log-mel spectrogram @ 22.05 kHz).
  3. Discrete Vector-Quantized bottleneck tokenization & bitrate analysis (~194 bps).
  4. Zero-shot voice conversion via FiLM speaker conditioning.
  5. Training objective loss breakdown (L1 Rec, VQ, GRL Leakage, Speaker Consistency).
  6. Chunked recurrent streaming inference with exact numerical verification.
  7. Real-Time Factor (RTF) and per-frame latency benchmarking on CPU and CUDA.
"""

import sys
import time
import torch
import torch.nn.functional as F

from ssm_codec.models.codec import SSMSpeechCodec
from ssm_codec.utils.audio import AudioProcessor, generate_synthetic_speech
from ssm_codec.utils.metrics import benchmark_rtf_and_latency, compute_cosine_similarity


def print_banner(text: str):
    line = "=" * 80
    print(f"\n{line}\n  {text}\n{line}")


def main():
    print_banner("IISc SPIRE Lab - SSM Speech Codec with Zero-Shot Speaker Conditioning")

    # 1. Hardware Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device.type.upper()} " + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 2. Model Initialization (Compact Research Configuration)
    print("\n[Architecture] Initializing SSM Speech Codec...")
    model = SSMSpeechCodec(
        mel_bins=80,
        model_dim=128,         # 128 for lightweight 1.5M budget (or 256 for 5M)
        d_state=32,
        num_layers=6,
        codebook_size=512,     # K=512 codebook entries (9 bits/frame)
        latent_dim=64,         # 64-d projection bottleneck
        spk_dim=192,           # 192-d ECAPA-TDNN speaker embedding
        num_speakers=50,
        commitment_weight=0.25,
        leakage_weight=0.1,
        consistency_weight=0.5,
    ).to(device)

    # 3. Parameter Count Audit
    counts = model.count_parameters()
    print_banner("Model Parameter Budget Breakdown")
    print(f"  {'Component':<30} | {'Parameters':>12}")
    print("  " + "-" * 45)
    print(f"  {'Content Encoder (Causal Conv + SSM)':<30} | {counts['encoder']:>12,}")
    print(f"  {'Vector Quantizer (K=512, D=64)':<30} | {counts['vq']:>12,}")
    print(f"  {'SSM Decoder (FiLM + Conv)':<30} | {counts['decoder']:>12,}")
    print(f"  {'Adversarial Speaker Classifier':<30} | {counts['speaker_classifier']:>12,}")
    print("  " + "-" * 45)
    print(f"  {'Total Trainable Codec Parameters':<30} | {counts['trainable_total']:>12,}")
    print(f"  {'Frozen ECAPA-TDNN Speaker Model':<30} | {counts['frozen_speaker_encoder']:>12,}")
    print(f"  {'Total Combined System Parameters':<30} | {counts['all_total']:>12,}")

    # 4. Bitrate & Information Bottleneck Analysis
    frame_rate = 22050.0 / (256.0 * 4.0)  # 22050 / (hop 256 * 4x downsample) ~ 21.53 Hz
    bits_per_token = 9.0  # log2(512) = 9 bits
    bitrate_bps = frame_rate * bits_per_token
    print_banner("Information Bottleneck & Bitrate Analysis")
    print(f"  Sampling Rate:                22,050 Hz")
    print(f"  Spectrogram Hop Size:         256 samples (~86.13 Hz)")
    print(f"  SSM Encoder Downsampling:     4x temporal compression")
    print(f"  Token Transmission Rate:      {frame_rate:.2f} tokens / second")
    print(f"  Codebook Cardinality:         512 entries ({bits_per_token:.0f} bits / token)")
    print(f"  Discrete Content Bitrate:     {bitrate_bps:.2f} bps (~0.19 kbps)")
    print(f"  Compression Ratio vs 16-bit:   {(22050 * 16) / bitrate_bps:.1f}x compression")

    # 5. Acoustic Input Generation (Two Distinct Speakers)
    print_banner("Generating Synthetic Speech Utterances")
    processor = AudioProcessor(sample_rate=22050, n_mels=80)

    # Speaker A: Source content utterance (Male voice, F0 ~ 125 Hz)
    wav_a, sr = generate_synthetic_speech(duration_sec=2.0, sample_rate=22050, speaker_f0=125.0, seed=101)
    mel_a = processor.wav_to_mel(wav_a.to(device)).to(device)  # (1, 80, time)

    # Speaker B: Target voice reference (Female voice, F0 ~ 230 Hz)
    wav_b, _ = generate_synthetic_speech(duration_sec=2.0, sample_rate=22050, speaker_f0=230.0, seed=202)
    mel_b = processor.wav_to_mel(wav_b.to(device)).to(device)  # (1, 80, time)

    print(f"  Source Speaker A (Male F0 ~125 Hz) Mel Shape:    {tuple(mel_a.shape)}")
    print(f"  Target Speaker B (Female F0 ~230 Hz) Mel Shape:  {tuple(mel_b.shape)}")

    # 6. Speaker Embedding Extraction (Frozen ECAPA-TDNN)
    spk_emb_a = model.extract_speaker_embedding(mel_a)
    spk_emb_b = model.extract_speaker_embedding(mel_b)
    initial_cosine = compute_cosine_similarity(spk_emb_a, spk_emb_b)
    print(f"  Speaker A vs Speaker B Cosine Similarity:       {initial_cosine:.4f} (distinct speakers)")

    # 7. End-to-End Forward Pass & Zero-Shot Voice Conversion
    print_banner("Executing Zero-Shot Voice Conversion Forward Pass")
    model.eval()
    with torch.no_grad():
        # Step A: Encode linguistic content from Speaker A
        z_q_a, indices_a, vq_metrics_a, _ = model.encode(mel_a)
        print(f"  Encoded Content Tokens Shape:  {tuple(indices_a.shape)} (quantized sequence length: {indices_a.shape[-1]})")
        print(f"  Active Codebook Perplexity:   {vq_metrics_a['perplexity']:.2f} / 512")

        # Step B: Decode with Speaker B conditioning via FiLM
        converted_mel, _ = model.decode(z_q_a, spk_emb=spk_emb_b)
        print(f"  Converted Mel Output Shape:    {tuple(converted_mel.shape)}")

        # Step C: Speaker Transfer Verification
        converted_spk_emb = model.extract_speaker_embedding(converted_mel)
        transfer_similarity = compute_cosine_similarity(converted_spk_emb, spk_emb_b)
        print(f"  Converted Output vs Target B Similarity: {transfer_similarity:.4f}")

    # 8. Training Loss Dynamics & Multitask Objectives
    print_banner("Training Objective Loss Breakdown (Step Simulation)")
    model.train()
    spk_id = torch.tensor([1], device=device)
    loss_out = model(mel=mel_a, spk_emb=spk_emb_a, spk_ids=spk_id, grl_alpha=1.0)
    losses = loss_out["losses"]
    print(f"  1. L1 Mel Reconstruction Loss:   {losses['loss_rec'].item():.4f}")
    print(f"  2. Vector Quantization Loss:     {losses['loss_vq'].item():.4f}")
    print(f"  3. Speaker Leakage Penalty:      {losses['loss_leak'].item():.4f} (Accuracy: {losses['leakage_accuracy'].item()*100:.1f}%)")
    print(f"  4. Speaker Consistency Loss:     {losses['loss_consistency'].item():.4f}")
    print(f"  ------------------------------------------------")
    print(f"  Total Aggregated Loss:           {loss_out['total_loss'].item():.4f}")

    # 9. Chunked Recurrent Streaming Inference Verification
    print_banner("Causal Chunked Streaming Equivalence Verification")
    model.eval()
    with torch.no_grad():
        # Offline full-batch output
        offline_mel, _ = model.decode(z_q_a, spk_emb=spk_emb_b)

        # A. Verify exact SSM sequence layer recurrence vs offline parallel FFT
        h_offline = model.decoder.in_proj(z_q_a.transpose(1, 2)).transpose(1, 2)
        for layer in model.decoder.layers:
            h_offline, _ = layer(h_offline, spk_emb=spk_emb_b, state=None)

        chunk_size = 4
        num_tokens = z_q_a.shape[-1]
        dec_states = model.decoder.init_states(batch_size=1, device=device)
        h_stream_chunks = []

        start_stream = time.perf_counter()
        for i in range(0, num_tokens, chunk_size):
            chunk_zq = z_q_a[:, :, i : i + chunk_size]
            h_c = model.decoder.in_proj(chunk_zq.transpose(1, 2)).transpose(1, 2)
            new_s = []
            for l_idx, layer in enumerate(model.decoder.layers):
                h_c, s = layer(h_c, spk_emb=spk_emb_b, state=dec_states[l_idx])
                new_s.append(s)
            dec_states = new_s
            h_stream_chunks.append(h_c)
        h_stream = torch.cat(h_stream_chunks, dim=-1)
        stream_time = time.perf_counter() - start_stream

        ssm_error = torch.abs(h_offline - h_stream).max().item()
        print(f"  Number of Streaming Chunks:    {len(h_stream_chunks)}")
        print(f"  SSM Recurrent Sequence Error:  {ssm_error:.2e} (tolerance < 1e-4: {'PASSED' if ssm_error < 1e-4 else 'FAILED'})")

        # B. Full Mel Output Equivalence
        stream_mel = model.decoder.out_proj(model.decoder.upsampler(h_stream))
        end_to_end_diff = torch.abs(offline_mel - stream_mel).max().item()
        print(f"  Full Mel Output Equivalence:   {end_to_end_diff:.2e} (tolerance < 1e-4: {'PASSED' if end_to_end_diff < 1e-4 else 'FAILED'})")

    # 10. Computational Efficiency & Latency Benchmark
    print_banner("Computational Efficiency Benchmark (RTF & Latency)")
    perf = benchmark_rtf_and_latency(
        model=model,
        mel=mel_a,
        spk_emb=spk_emb_b,
        warmup=5,
        runs=25,
    )
    print(f"  Audio Utterance Duration:      {perf['audio_duration_sec']:.2f} seconds")
    print(f"  Mean Codec Processing Time:    {perf['avg_compute_sec']*1000.0:.2f} ms")
    print(f"  Real-Time Factor (RTF):        {perf['rtf']:.4f} (RTF < 1.0 means faster than real-time)")
    print(f"  Per-Frame Latency:             {perf['per_frame_latency_ms']:.3f} ms / frame")
    print(f"  Real-Time Speedup:             {1.0 / perf['rtf']:.1f}x real-time speed")

    print_banner("Research Demonstration Completed Successfully")


if __name__ == "__main__":
    main()
