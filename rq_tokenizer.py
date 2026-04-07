"""
RQ-KMeans+ Tokenizer for GPR.

Converts item embeddings into hierarchical semantic IDs via residual
quantization. Combines the strengths of RQ-KMeans (high codebook utilization
via K-means initialization) with RQ-VAE (learnable latent space via
encoder-decoder fine-tuning with residual connections).

Architecture:
  1. Encoder with residual connection: z = Encoder(x) + x
  2. Multi-level residual quantization with K-means-initialized codebooks
  3. Decoder: x_hat = Decoder(quantized)
  4. Loss = reconstruction + commitment
"""

import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from config import TokenizerConfig


class ResidualEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.shortcut = (
            nn.Linear(input_dim, embed_dim)
            if input_dim != embed_dim
            else nn.Identity()
        )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RQKMeansPlus(nn.Module):
    """
    RQ-KMeans+ quantizer.

    Phase 1 (fit_kmeans): Initialize codebooks with RQ-KMeans clustering.
    Phase 2 (fit_vae):    Fine-tune encoder, codebooks, decoder end-to-end.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.n_levels = cfg.n_levels
        self.codebook_size = cfg.codebook_size
        self.embed_dim = cfg.embed_dim
        self.input_dim = cfg.input_dim

        self.encoder = ResidualEncoder(cfg.input_dim, cfg.embed_dim)
        self.decoder = Decoder(cfg.embed_dim, cfg.input_dim)

        # Codebooks: one per level
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(cfg.codebook_size, cfg.embed_dim) * 0.01)
            for _ in range(cfg.n_levels)
        ])

        self._fitted = False

    def _quantize_level(self, residual, level):
        """Find nearest codebook entry and compute quantized + new residual."""
        cb = self.codebooks[level]  # [C, D]
        # dist[i,j] = ||residual[i] - cb[j]||^2
        dist = (
            residual.pow(2).sum(dim=-1, keepdim=True)
            - 2 * residual @ cb.t()
            + cb.pow(2).sum(dim=-1, keepdim=True).t()
        )
        indices = dist.argmin(dim=-1)  # [B]
        quantized = cb[indices]        # [B, D]
        new_residual = residual - quantized.detach()
        return indices, quantized, new_residual

    def encode(self, x):
        """
        Encode input embeddings to semantic IDs.
        x: [B, input_dim]
        Returns: codes [B, n_levels], quantized [B, embed_dim]
        """
        z = self.encoder(x)
        residual = z
        codes = []
        quantized_sum = torch.zeros_like(z)

        for level in range(self.n_levels):
            indices, quantized, residual = self._quantize_level(residual, level)
            codes.append(indices)
            quantized_sum = quantized_sum + quantized

        codes = torch.stack(codes, dim=-1)  # [B, n_levels]
        return codes, quantized_sum, z

    def decode(self, quantized):
        """Decode quantized representation back to input space."""
        return self.decoder(quantized)

    def forward(self, x):
        """Full forward: encode -> quantize -> decode."""
        codes, quantized_sum, z = self.encode(x)
        x_hat = self.decode(quantized_sum)

        # Losses
        recon_loss = F.mse_loss(x_hat, x)
        commit_loss = F.mse_loss(z, quantized_sum.detach())

        # Straight-through gradient for quantization
        quantized_st = z + (quantized_sum - z).detach()
        x_hat_st = self.decode(quantized_st)
        recon_loss_st = F.mse_loss(x_hat_st, x)

        total_loss = recon_loss_st + 0.25 * commit_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.item(),
            "commit_loss": commit_loss.item(),
            "codes": codes,
            "x_hat": x_hat,
        }

    @torch.no_grad()
    def fit_kmeans(self, embeddings: np.ndarray):
        """
        Phase 1: Initialize codebooks with RQ-KMeans.
        embeddings: [N, input_dim] numpy array
        """
        print("Phase 1: Initializing codebooks with RQ-KMeans...")
        device = next(self.parameters()).device

        # Project through encoder (randomly initialized)
        x = torch.tensor(embeddings, dtype=torch.float32, device=device)
        z = self.encoder(x).cpu().numpy()

        residual = z.copy()
        for level in range(self.n_levels):
            n_clusters = min(self.codebook_size, len(residual))
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(4096, len(residual)),
                n_init=3,
                random_state=42 + level,
            )
            kmeans.fit(residual)
            centers = kmeans.cluster_centers_
            cb_data = self.codebooks[level].data.clone()
            cb_data[:n_clusters] = torch.tensor(
                centers, dtype=torch.float32, device=device
            )
            self.codebooks[level].data = cb_data

            # Compute residual for next level
            labels = kmeans.predict(residual)
            residual = residual - centers[labels]
            usage = len(np.unique(labels)) / self.codebook_size
            print(f"  Level {level}: codebook usage = {usage:.1%}")

        self._fitted = True
        print("Phase 1 complete.")

    def fit(self, embeddings: np.ndarray, cfg: TokenizerConfig):
        """Full training: KMeans init + VAE fine-tuning."""
        device = next(self.parameters()).device

        self.fit_kmeans(embeddings)

        print("Phase 2: Fine-tuning with encoder-decoder...")
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        dataset = torch.tensor(embeddings, dtype=torch.float32)
        n = len(dataset)

        for epoch in range(cfg.epochs):
            perm = torch.randperm(n)
            total_loss = 0
            n_batches = 0

            for i in range(0, n, cfg.batch_size):
                batch = dataset[perm[i:i + cfg.batch_size]].to(device)
                result = self.forward(batch)
                loss = result["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg = total_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{cfg.epochs} | loss={avg:.4f}")

        # Compute final metrics
        self._compute_metrics(embeddings)
        print("Phase 2 complete.")

    @torch.no_grad()
    def _compute_metrics(self, embeddings: np.ndarray):
        """Compute collision rate, codebook usage, and PAS (Table 1)."""
        device = next(self.parameters()).device
        x = torch.tensor(embeddings, dtype=torch.float32, device=device)

        all_codes = []
        bs = 4096
        for i in range(0, len(x), bs):
            codes, _, _ = self.encode(x[i:i + bs])
            all_codes.append(codes.cpu())
        all_codes = torch.cat(all_codes, dim=0).numpy()

        # Collision rate: fraction of items sharing same full code
        code_strs = ["_".join(map(str, c)) for c in all_codes]
        code_counts = Counter(code_strs)
        n_collisions = sum(v - 1 for v in code_counts.values() if v > 1)
        collision_rate = n_collisions / len(code_strs)

        # Code usage at level 1
        unique_l1 = len(set(all_codes[:, 0].tolist()))
        cur_l1 = unique_l1 / self.codebook_size

        # Path Average Similarity (PAS): mean cosine similarity among items
        # sharing the same full semantic code. Higher PAS means collisions
        # group semantically coherent items (Table 1 in paper).
        code_to_indices = defaultdict(list)
        for i, cs in enumerate(code_strs):
            code_to_indices[cs].append(i)

        total_pas = 0.0
        n_collision_groups = 0
        for indices in code_to_indices.values():
            if len(indices) < 2:
                continue
            group_embs = embeddings[indices]
            norms = np.linalg.norm(group_embs, axis=1, keepdims=True) + 1e-8
            normalized = group_embs / norms
            sim_matrix = normalized @ normalized.T
            n = len(indices)
            avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
            total_pas += avg_sim
            n_collision_groups += 1

        pas = total_pas / max(n_collision_groups, 1)

        print(f"  Collision Rate: {collision_rate:.2%}")
        print(f"  Code Usage (L1): {cur_l1:.2%}")
        print(f"  Path Average Similarity (PAS): {pas:.4f}")

    @torch.no_grad()
    def encode_all(self, embeddings: np.ndarray, batch_size: int = 4096):
        """Encode all items and return semantic IDs as numpy array."""
        device = next(self.parameters()).device
        all_codes = []
        for i in range(0, len(embeddings), batch_size):
            x = torch.tensor(
                embeddings[i:i + batch_size], dtype=torch.float32, device=device
            )
            codes, _, _ = self.encode(x)
            all_codes.append(codes.cpu().numpy())
        return np.concatenate(all_codes, axis=0)

    def save(self, path: str):
        # Always save as CPU tensors for cross-device portability (CUDA ↔ MUSA)
        torch.save({k: v.cpu() for k, v in self.state_dict().items()}, path)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        self._fitted = True
        print(f"Tokenizer loaded from {path}")
