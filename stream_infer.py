"""Interactive Streaming Inference Demo with Causal State Caching."""

import argparse
import time
import torch
import soundfile as sf

from ssm_codec.models.codec import SSMSpeechCodec
from ssm_codec.utils.audio import AudioProcessor, generate_synthetic_speech


def stream_infer(chunk_tokens: int = 4, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    processor = AudioProcessor(sample_rate=22050, n_mels=80)
    wav_src, _ = generate_synthetic_speech(duration_sec=3.0, speaker_f0=130.0, seed=12)
    wav_tgt, _ = generate_synthetic_speech(duration_sec=2.0, speaker_f0=220.0, seed=99)

    mel_src = processor.wav_to_mel(wav_src.to(device)).to(device)
    mel_tgt = processor.wav_to_mel(wav_tgt.to(device)).to(device)

    # Pad source to multiple of 4
    t_len = mel_src.shape[-1]
    pad = (4 - (t_len % 4)) % 4
    if pad > 0:
        mel_src = torch.nn.functional.pad(mel_src, (0, pad))

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

    with torch.no_grad():
        # Pre-extract target speaker conditioning
        spk_emb = model.extract_speaker_embedding(mel_tgt)

        # Full offline encode of source content into VQ tokens
        z_q, indices, _, _ = model.encode(mel_src)
        total_tokens = z_q.shape[-1]

        # Initialize recurrent decoder states
        dec_states = model.decoder.init_states(batch_size=1, device=device)

        print(f"[Streaming Demo] Streaming {total_tokens} tokens in chunks of {chunk_tokens} tokens...")
        print(f"  {'Chunk':<6} | {'Tokens':<10} | {'Latency (ms)':>14} | {'Status':<10}")
        print("  " + "-" * 48)

        latencies = []
        streamed_mel_chunks = []

        for c_idx, i in enumerate(range(0, total_tokens, chunk_tokens)):
            chunk_zq = z_q[:, :, i : i + chunk_tokens]
            actual_chunk_len = chunk_zq.shape[-1]

            t0 = time.perf_counter()
            mel_chunk, dec_states = model.decode(chunk_zq, spk_emb=spk_emb, ssm_states=dec_states)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)
            streamed_mel_chunks.append(mel_chunk)

            print(f"  #{c_idx+1:<5} | {f'{i}:{i+actual_chunk_len}':<10} | {t_ms:>14.2f} | OK")

        avg_lat = sum(latencies) / len(latencies)
        print("  " + "-" * 48)
        print(f"[Summary] Mean chunk decoding latency: {avg_lat:.2f} ms")
        chunk_duration_sec = (chunk_tokens * 4 * 256) / 22050.0
        print(f"[Summary] Audio represented per chunk: {chunk_duration_sec*1000.0:.1f} ms")
        print(f"[Summary] Streaming Real-Time Factor:  {avg_lat / (chunk_duration_sec * 1000.0):.4f} (<< 1.0 Real-time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming inference demo")
    parser.add_argument("--chunk-tokens", type=int, default=4, help="Tokens per chunk")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    stream_infer(args.chunk_tokens, args.device)
