"""End-to-end SSM Speech Codec with Zero-Shot Speaker Conditioning."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm_codec.models.decoder import SSMDecoder
from ssm_codec.models.ecapa_tdnn import ECAPATDNN
from ssm_codec.models.encoder import ContentEncoder
from ssm_codec.modules.grl import SpeakerClassifier
from ssm_codec.modules.vq import VectorQuantizer


class SSMSpeechCodec(nn.Module):
    """Full SSM Speech Codec architecture integrating:

    1. Content Encoder (Causal Conv 4x downsampler + 6-layer SSM + 64-d projection)
    2. Vector Quantizer (K=512, D=64)
    3. Speaker Encoder (Frozen 192-d ECAPA-TDNN)
    4. Decoder (FiLM conditioning + 6-layer SSM + Causal Conv 4x upsampler)
    5. Speaker Classifier with Gradient Reversal Layer (Adversarial disentanglement)
    """

    def __init__(
        self,
        mel_bins: int = 80,
        model_dim: int = 256,
        d_state: int = 64,
        num_layers: int = 6,
        codebook_size: int = 512,
        latent_dim: int = 64,
        spk_dim: int = 192,
        num_speakers: int = 100,
        commitment_weight: float = 0.25,
        leakage_weight: float = 0.1,
        consistency_weight: float = 0.5,
    ):
        super().__init__()
        self.mel_bins = mel_bins
        self.model_dim = model_dim
        self.d_state = d_state
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.spk_dim = spk_dim
        self.leakage_weight = leakage_weight
        self.consistency_weight = consistency_weight

        # 1. Content Encoder
        self.encoder = ContentEncoder(
            in_channels=mel_bins,
            model_dim=model_dim,
            d_state=d_state,
            num_layers=num_layers,
            latent_dim=latent_dim,
        )

        # 2. Vector Quantizer Bottleneck
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_weight,
        )

        # 3. Speaker Conditioning Encoder (Frozen)
        self.speaker_encoder = ECAPATDNN(
            in_channels=mel_bins,
            channels=model_dim,
            emb_dim=spk_dim,
        )
        # Freeze speaker encoder
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

        # 4. Decoder with FiLM modulation
        self.decoder = SSMDecoder(
            latent_dim=latent_dim,
            model_dim=model_dim,
            d_state=d_state,
            num_layers=num_layers,
            spk_dim=spk_dim,
            out_channels=mel_bins,
        )

        # 5. Adversarial Speaker Classifier
        self.speaker_classifier = SpeakerClassifier(
            in_dim=latent_dim,
            hidden_dim=128,
            num_speakers=num_speakers,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Speaker encoder is permanently frozen
        self.speaker_encoder.eval()
        return self

    def extract_speaker_embedding(self, ref_mel: torch.Tensor) -> torch.Tensor:
        """Extracts frozen speaker embedding from reference mel spectrogram.

        Args:
            ref_mel: (batch, mel_bins, time)

        Returns:
            spk_emb: (batch, spk_dim)
        """
        self.speaker_encoder.eval()
        with torch.no_grad():
            return self.speaker_encoder(ref_mel)

    def init_streaming_states(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Initializes both encoder and decoder recurrent states for chunked streaming."""
        enc_states = self.encoder.init_states(batch_size, device)
        dec_states = self.decoder.init_states(batch_size, device)
        return enc_states, dec_states

    def encode(
        self,
        mel: torch.Tensor,
        ssm_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        """Encodes mel spectrogram into discrete content tokens.

        Args:
            mel: (batch, mel_bins, time)
            ssm_states: Optional encoder states for streaming

        Returns:
            z_q: Quantized embeddings (batch, latent_dim, time // 4)
            indices: Discrete codebook tokens (batch, time // 4)
            vq_metrics: VQ loss and perplexity dict
            new_states: Updated states if streaming
        """
        z_e, new_states = self.encoder(mel, ssm_states=ssm_states)
        z_q, indices, vq_metrics = self.vq(z_e)
        return z_q, indices, vq_metrics, new_states

    def decode(
        self,
        z_q: torch.Tensor,
        spk_emb: torch.Tensor,
        ssm_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Decodes quantized tokens into mel spectrogram conditioned on speaker embedding.

        Args:
            z_q: Quantized embeddings (batch, latent_dim, time // 4)
            spk_emb: Speaker embedding (batch, spk_dim)
            ssm_states: Optional decoder states for streaming

        Returns:
            mel_pred: Reconstructed mel spectrogram (batch, mel_bins, time)
            new_states: Updated decoder states if streaming
        """
        return self.decoder(z_q, spk_emb=spk_emb, states=ssm_states)

    def forward(
        self,
        mel: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
        ref_mel: Optional[torch.Tensor] = None,
        spk_ids: Optional[torch.Tensor] = None,
        grl_alpha: float = 1.0,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Full training forward pass computing all 4 objective terms.

        Args:
            mel: Input mel spectrogram (batch, mel_bins, time)
            spk_emb: Optional pre-extracted speaker embedding (batch, spk_dim)
            ref_mel: Optional reference mel spectrogram (if spk_emb is not provided)
            spk_ids: Ground-truth speaker IDs (batch,) for adversarial leakage loss
            grl_alpha: Gradient reversal scale for current training step

        Returns:
            Dictionary containing:
                - mel_pred: Reconstructed mel spectrogram
                - indices: Codebook indices
                - losses: Individual loss terms (L1 rec, VQ, leakage, consistency)
                - total_loss: Aggregated scalar loss for backprop
        """
        # Ensure mel time dimension is a multiple of downsampling factor (4)
        time_len = mel.shape[-1]
        pad_len = (4 - (time_len % 4)) % 4
        if pad_len > 0:
            mel = F.pad(mel, (0, pad_len))

        # 1. Obtain speaker embedding
        if spk_emb is None:
            source_for_spk = ref_mel if ref_mel is not None else mel
            spk_emb = self.extract_speaker_embedding(source_for_spk)

        # 2. Content Encoding & Vector Quantization
        z_q, indices, vq_metrics, _ = self.encode(mel)

        # 3. Decoding conditioned on target speaker
        mel_pred, _ = self.decode(z_q, spk_emb=spk_emb)

        # Remove pad if added
        if pad_len > 0:
            mel = mel[..., :time_len]
            mel_pred = mel_pred[..., :time_len]

        # ---------------------------------------------------------------------
        # 4. Objective Functions
        # ---------------------------------------------------------------------
        # Term 1: Mel L1 Reconstruction Loss
        loss_rec = F.l1_loss(mel_pred, mel)

        # Term 2: Quantization Loss (weighted at 0.25 as per proposal)
        loss_vq = vq_metrics["vq_loss"]

        # Term 3: Speaker-leakage penalty via GRL & speaker classifier
        if spk_ids is not None:
            spk_logits = self.speaker_classifier(z_q, alpha=grl_alpha)
            loss_leak = F.cross_entropy(spk_logits, spk_ids)
            # Accuracy metric
            with torch.no_grad():
                spk_pred = torch.argmax(spk_logits, dim=-1)
                leak_acc = (spk_pred == spk_ids).float().mean()
        else:
            loss_leak = torch.tensor(0.0, device=mel.device)
            leak_acc = torch.tensor(0.0, device=mel.device)

        # Term 4: Speaker Consistency Loss (Cosine loss between reconstructed & target)
        with torch.no_grad():
            rec_spk_emb = self.speaker_encoder(mel_pred)
        # Cosine distance: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(rec_spk_emb, spk_emb, dim=-1).mean()
        loss_consistency = 1.0 - cos_sim

        # Total Aggregated Loss
        total_loss = (
            loss_rec
            + 0.25 * loss_vq
            + self.leakage_weight * loss_leak
            + self.consistency_weight * loss_consistency
        )

        return {
            "mel_pred": mel_pred,
            "indices": indices,
            "total_loss": total_loss,
            "losses": {
                "loss_rec": loss_rec,
                "loss_vq": loss_vq,
                "loss_leak": loss_leak,
                "loss_consistency": loss_consistency,
                "cosine_similarity": cos_sim,
                "leakage_accuracy": leak_acc,
                "perplexity": vq_metrics["perplexity"],
                "codebook_usage": vq_metrics["usage"],
            },
        }

    def count_parameters(self) -> Dict[str, int]:
        """Calculates detailed parameter counts across components."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        vq_params = sum(p.numel() for p in self.vq.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        spk_cls_params = sum(p.numel() for p in self.speaker_classifier.parameters())
        spk_enc_params = sum(p.numel() for p in self.speaker_encoder.parameters())

        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_total = sum(p.numel() for p in self.parameters())

        return {
            "encoder": enc_params,
            "vq": vq_params,
            "decoder": dec_params,
            "speaker_classifier": spk_cls_params,
            "frozen_speaker_encoder": spk_enc_params,
            "trainable_total": trainable_total,
            "all_total": all_total,
        }
