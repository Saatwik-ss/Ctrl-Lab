"""Vector Quantization (VQ) bottleneck with Straight-Through Estimator.

Implements discrete representation learning with commitment loss and
perplexity tracking for measuring codebook usage.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantizer with Straight-Through Estimator and Commitment Loss.

    Args:
        num_embeddings: Codebook size (K=512)
        embedding_dim: Latent vector dimension (D=64)
        commitment_cost: Commitment loss weight beta (default 0.25)
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self,
        z_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantizes continuous latents z_e.

        Args:
            z_e: Continuous encoder latents (batch, embedding_dim, time)

        Returns:
            z_q: Quantized representation with straight-through gradients (batch, embedding_dim, time)
            indices: Discrete codebook indices (batch, time)
            metrics: Dictionary containing vq_loss, codebook_loss, commitment_loss, perplexity, and usage
        """
        # Permute to (batch, time, embedding_dim)
        z_e_perm = z_e.permute(0, 2, 1).contiguous()
        flat_z_e = z_e_perm.view(-1, self.embedding_dim)  # (B * T, D)

        # Compute Euclidean distance: ||z_e - e_j||^2 = ||z_e||^2 + ||e_j||^2 - 2 z_e . e_j
        distances = (
            torch.sum(flat_z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z_e, self.embedding.weight.t())
        )

        # Nearest neighbor index selection
        indices_flat = torch.argmin(distances, dim=1)
        indices = indices_flat.view(z_e_perm.shape[0], z_e_perm.shape[1])

        # Retrieve quantized vectors
        z_q_flat = self.embedding(indices_flat)
        z_q = z_q_flat.view(z_e_perm.shape)

        # Compute VQ losses
        # Codebook loss: moves embeddings towards encoder latents
        loss_codebook = F.mse_loss(z_q, z_e_perm.detach())
        # Commitment loss: encourages encoder to commit to codebook vectors
        loss_commitment = F.mse_loss(z_e_perm, z_q.detach())
        # Total VQ loss
        vq_loss = loss_codebook + self.commitment_cost * loss_commitment

        # Straight-Through Estimator (STE)
        # z_q_st has the values of z_q, but the gradients of z_e_perm
        z_q_st = z_e_perm + (z_q - z_e_perm).detach()

        # Permute back to (batch, embedding_dim, time)
        z_q_out = z_q_st.permute(0, 2, 1).contiguous()

        # Compute diagnostics: Perplexity and codebook usage
        with torch.no_grad():
            # One-hot encoding of assignments
            encodings = F.one_hot(indices_flat, self.num_embeddings).float()
            probs = torch.mean(encodings, dim=0)  # (K,)
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))
            used_codes = torch.sum(probs > 0).float()
            usage_ratio = used_codes / self.num_embeddings

        metrics = {
            "vq_loss": vq_loss,
            "codebook_loss": loss_codebook,
            "commitment_loss": loss_commitment,
            "perplexity": perplexity,
            "usage": usage_ratio,
        }

        return z_q_out, indices, metrics

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decodes codebook indices into quantized embeddings.

        Args:
            indices: (batch, time) integer tensor

        Returns:
            z_q: (batch, embedding_dim, time)
        """
        z_q = self.embedding(indices)  # (batch, time, D)
        return z_q.permute(0, 2, 1).contiguous()
