"""Training pipeline for SSM Speech Codec with Zero-Shot Speaker Conditioning."""

import argparse
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from ssm_codec.models.codec import SSMSpeechCodec
from ssm_codec.utils.audio import AudioProcessor, generate_synthetic_speech


class SyntheticSpeechDataset(Dataset):
    """Synthetic speech dataset for reproducible testing and fast training verification."""

    def __init__(self, num_samples: int = 200, duration_sec: float = 2.0, num_speakers: int = 50):
        self.num_samples = num_samples
        self.duration_sec = duration_sec
        self.num_speakers = num_speakers
        self.processor = AudioProcessor(sample_rate=22050, n_mels=80)

        # Pre-generate diverse speaker fundamental frequencies
        self.speaker_f0s = [100.0 + (i * 180.0 / num_speakers) for i in range(num_speakers)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        spk_id = idx % self.num_speakers
        f0 = self.speaker_f0s[spk_id]
        wav, _ = generate_synthetic_speech(
            duration_sec=self.duration_sec,
            sample_rate=22050,
            speaker_f0=f0,
            seed=idx + 1000,
        )
        mel = self.processor.wav_to_mel(wav)  # (1, 80, time)
        return {
            "mel": mel.squeeze(0),         # (80, time)
            "spk_id": torch.tensor(spk_id, dtype=torch.long),
        }


def collate_fn(batch):
    mels = [item["mel"] for item in batch]
    spk_ids = torch.stack([item["spk_id"] for item in batch])

    # Find maximum length and pad to multiple of 4
    max_len = max(m.shape[-1] for m in mels)
    pad_to = ((max_len + 3) // 4) * 4

    padded_mels = []
    for m in mels:
        pad_amount = pad_to - m.shape[-1]
        padded = torch.nn.functional.pad(m, (0, pad_amount))
        padded_mels.append(padded)

    batch_mels = torch.stack(padded_mels, dim=0)  # (batch, 80, time)
    return {
        "mel": batch_mels,
        "spk_id": spk_ids,
    }


def train(config_path: str, max_epochs: int = 3, device_str: str = "auto"):
    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[Train] Initializing training on {device}...")

    # Build model
    model = SSMSpeechCodec(
        mel_bins=config["model"]["mel_bins"],
        model_dim=config["model"]["model_dim"],
        d_state=config["model"].get("d_state", 64),
        num_layers=config["model"]["num_layers"],
        codebook_size=config["model"]["codebook_size"],
        latent_dim=config["model"]["latent_dim"],
        spk_dim=config["model"]["spk_dim"],
        num_speakers=config["model"]["num_speakers"],
        commitment_weight=config["loss"]["commitment_weight"],
        leakage_weight=config["loss"]["leakage_weight"],
        consistency_weight=config["loss"]["consistency_weight"],
    ).to(device)

    # Optimizer with AdamW and cosine decay
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 0.01))
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    dataset = SyntheticSpeechDataset(num_samples=120, duration_sec=1.5, num_speakers=config["model"]["num_speakers"])
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * len(dataloader), eta_min=1e-6)

    model.train()
    print(f"[Train] Dataset loaded with {len(dataset)} samples. Starting {max_epochs} epochs...")

    step = 0
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0.0
        start_epoch = time.time()

        for batch_idx, batch in enumerate(dataloader):
            step += 1
            mel = batch["mel"].to(device)
            spk_id = batch["spk_id"].to(device)

            optimizer.zero_grad()
            out = model(mel=mel, spk_ids=spk_id, grl_alpha=1.0)
            loss = out["total_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["training"].get("grad_clip", 1.0)))
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 5 == 0 or batch_idx == len(dataloader) - 1:
                losses = out["losses"]
                print(
                    f"Epoch {epoch:02d} [{batch_idx+1:02d}/{len(dataloader):02d}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Rec: {losses['loss_rec'].item():.4f} | "
                    f"VQ: {losses['loss_vq'].item():.4f} | "
                    f"Leak: {losses['loss_leak'].item():.4f} | "
                    f"Perp: {losses['perplexity'].item():.1f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"--> Epoch {epoch:02d} completed in {time.time() - start_epoch:.2f}s | Avg Loss: {avg_epoch_loss:.4f}\n")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/ssm_codec_latest.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
    }, ckpt_path)
    print(f"[Train] Model checkpoint successfully saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSM Speech Codec")
    parser.add_argument("--config", type=str, default="configs/default_ssm.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")
    args = parser.parse_args()

    train(args.config, max_epochs=args.epochs, device_str=args.device)
