"""
GPR Training Pipeline with TensorBoard Monitoring.

Three-stage training:
  Stage 1: Pre-training with Multi-Token Prediction (MTP)
  Stage 2: Value-Aware Fine-Tuning (VAFT)
  Stage 3: Post-training with HEPO (Hierarchical Enhanced Policy Optimization)

All losses, metrics, and learning rates are logged to TensorBoard.

Usage:
  # Full pipeline with synthetic data (auto-generated, no download)
  python train.py --dataset synthetic

  # Full pipeline with Amazon Reviews (auto-downloaded)
  python train.py --dataset amazon

  # Single stage
  python train.py --stage mtp
  python train.py --stage vaft --resume checkpoints/mtp_best.pt
  python train.py --stage hepo --resume checkpoints/vaft_best.pt

  # Launch TensorBoard (in another terminal)
  tensorboard --logdir runs/ --port 6006
"""

import os
import argparse
import random
import time
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import GPRConfig
from data_utils import (
    generate_synthetic_data,
    load_amazon_reviews,
    build_sequences,
    create_dataloaders,
    build_user_code_popularity,
    get_all_codes_per_level,
    generate_arr_samples,
    merge_batches,
)
from rq_tokenizer import RQKMeansPlus
from model import GPR, SemanticTrie, mtp_loss, vaft_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# TensorBoard helper
# ---------------------------------------------------------------------------

class TBLogger:
    """Wraps TensorBoard SummaryWriter with step tracking."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def log_scalars(self, tag_prefix: str, scalars: dict, step: int = None):
        s = step if step is not None else self.global_step
        for k, v in scalars.items():
            self.writer.add_scalar(f"{tag_prefix}/{k}", v, s)

    def log_scalar(self, tag: str, value: float, step: int = None):
        s = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, s)

    def step(self):
        self.global_step += 1

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


# ---------------------------------------------------------------------------
# Stage 1: Pre-training with Multi-Token Prediction
# ---------------------------------------------------------------------------

def train_mtp(model, train_loader, val_loader, cfg, tb: TBLogger):
    print("\n" + "=" * 60)
    print("Stage 1: Pre-training with Multi-Token Prediction (MTP)")
    print("=" * 60)

    device = cfg.train.device
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.mtp_lr,
        weight_decay=cfg.train.mtp_weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.mtp_epochs, eta_min=cfg.train.mtp_lr * 0.01
    )

    best_val_loss = float("inf")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    for epoch in range(cfg.train.mtp_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {"ce_loss": 0.0, "refine_loss": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"MTP Epoch {epoch+1}/{cfg.train.mtp_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            result = model(batch, mode="mtp")
            loss, metrics = mtp_loss(
                result, batch["target_ids"], n_heads=cfg.model.n_mtp_heads
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            tb.log_scalars("mtp_train_step", {
                "total_loss": loss.item(),
                "ce_loss": metrics["ce_loss"],
                "refine_loss": metrics["refine_loss"],
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
            })
            tb.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        val_loss, val_metrics = evaluate_model(model, val_loader, cfg, mode="mtp")

        tb.log_scalars("mtp_train_epoch", {
            "avg_loss": avg_loss,
            **{f"avg_{k}": v for k, v in avg_metrics.items()},
        }, step=epoch)
        tb.log_scalars("mtp_val_epoch", {
            "loss": val_loss,
            "hitrate_l1": val_metrics["hitrate_l1"],
            "hitrate_full": val_metrics["hitrate_full"],
        }, step=epoch)
        tb.log_scalar("lr/mtp", optimizer.param_groups[0]["lr"], step=epoch)
        tb.flush()

        print(
            f"  Train Loss: {avg_loss:.4f} | "
            f"CE: {avg_metrics['ce_loss']:.4f} | "
            f"Refine: {avg_metrics['refine_loss']:.4f}"
        )
        print(
            f"  Val Loss:   {val_loss:.4f} | "
            f"HitRate@L1: {val_metrics['hitrate_l1']:.4f} | "
            f"HitRate@Full: {val_metrics['hitrate_full']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg.train.save_dir, "mtp_best.pt"))
            print("  -> Saved best MTP model")

    print(f"MTP training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 2: Value-Aware Fine-Tuning (VAFT)
# ---------------------------------------------------------------------------

def train_vaft(model, train_loader, val_loader, cfg, tb: TBLogger):
    print("\n" + "=" * 60)
    print("Stage 2: Value-Aware Fine-Tuning (VAFT)")
    print("=" * 60)

    device = cfg.train.device
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.vaft_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.vaft_epochs, eta_min=cfg.train.vaft_lr * 0.01
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.train.vaft_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {"ce_loss": 0.0, "value_loss": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"VAFT Epoch {epoch+1}/{cfg.train.vaft_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            result = model(batch, mode="vaft")
            loss, metrics = vaft_loss(
                result, batch["target_ids"],
                batch["target_value"], batch["target_action"],
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            tb.log_scalars("vaft_train_step", {
                "total_loss": loss.item(),
                "ce_loss": metrics["ce_loss"],
                "value_loss": metrics["value_loss"],
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
            })
            tb.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        val_loss, val_metrics = evaluate_model(model, val_loader, cfg, mode="vaft")

        tb.log_scalars("vaft_train_epoch", {
            "avg_loss": avg_loss,
            **{f"avg_{k}": v for k, v in avg_metrics.items()},
        }, step=epoch)
        tb.log_scalars("vaft_val_epoch", {
            "loss": val_loss,
            "hitrate_l1": val_metrics["hitrate_l1"],
            "hitrate_full": val_metrics["hitrate_full"],
        }, step=epoch)
        tb.log_scalar("lr/vaft", optimizer.param_groups[0]["lr"], step=epoch)
        tb.flush()

        print(
            f"  Train Loss: {avg_loss:.4f} | "
            f"CE: {avg_metrics['ce_loss']:.4f} | "
            f"Value: {avg_metrics['value_loss']:.4f}"
        )
        print(
            f"  Val Loss:   {val_loss:.4f} | "
            f"HitRate@L1: {val_metrics['hitrate_l1']:.4f} | "
            f"HitRate@Full: {val_metrics['hitrate_full']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg.train.save_dir, "vaft_best.pt"))
            print("  -> Saved best VAFT model")

    print(f"VAFT training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 3: Post-training with HEPO
# ---------------------------------------------------------------------------

def _compute_batch_popularity(semantic_ids, token_types, action_types, n_levels):
    """
    On-the-fly per-sample code popularity from positive I-token interactions
    within the current batch (Eq. 6).
    """
    B = semantic_ids.shape[0]
    popularity = []

    for b in range(B):
        i_mask = (token_types[b] == 3)
        i_positions = i_mask.nonzero(as_tuple=False).squeeze(-1)
        n_i = len(i_positions)

        actions_b = action_types[b][:n_i]

        pop_per_level = {}
        for lvl in range(n_levels):
            codes = semantic_ids[b, i_positions, lvl]
            positive = (actions_b > 0)
            pos_codes = codes[positive]

            if len(pos_codes) == 0:
                pop_per_level[lvl] = {}
                continue

            unique_codes, counts = pos_codes.unique(return_counts=True)
            total = counts.sum().float()
            pop_per_level[lvl] = {
                c.item(): (cnt / total).item()
                for c, cnt in zip(unique_codes, counts)
            }

        popularity.append(pop_per_level)

    return popularity


def compute_process_rewards(codes, popularity, all_codes_per_level,
                            terminal_rewards, alpha=0.1):
    """
    Popularity-based hierarchical process rewards (Eq. 6–7).

    For ℓ < L:
        Δ_ℓ = P_ℓ(z_ℓ) − avg P_ℓ(t)  for t ∈ S_ℓ
        r_ℓ = α_ℓ · max(0, Δ_ℓ)
    For ℓ = L:
        r_L = R  (terminal reward from simulator)
    """
    B, K, n_levels = codes.shape
    device = codes.device
    rewards = torch.zeros(B, K, n_levels, device=device)

    for b in range(B):
        for lvl in range(n_levels):
            if lvl < n_levels - 1:
                pop_b = popularity[b].get(lvl, {})
                legal_codes = all_codes_per_level.get(lvl, set())
                avg_pop = (
                    sum(pop_b.get(c, 0.0) for c in legal_codes)
                    / max(len(legal_codes), 1)
                )

                for k_idx in range(K):
                    code_val = codes[b, k_idx, lvl].item()
                    p_code = pop_b.get(code_val, 0.0)
                    delta = p_code - avg_pop
                    rewards[b, k_idx, lvl] = alpha * max(0.0, delta)
            else:
                rewards[b, :, lvl] = terminal_rewards[b, :]

    return rewards


def train_hepo(model, train_loader, val_loader, cfg, tb: TBLogger,
               extra_data=None):
    """
    HEPO training with:
      - Popularity-based process rewards  (Eq. 6–7)
      - Full GAE with λ                   (Eq. 8)
      - Per-level coefficients c_ℓ        (Eq. 9)
      - Per-level value loss              (Eq. 10)
      - Anticipatory Request Rehearsal     (Sec. 3.3)
    """
    print("\n" + "=" * 60)
    print("Stage 3: Post-training with HEPO")
    print("=" * 60)

    device = cfg.train.device
    model = model.to(device)

    n_levels = cfg.model.n_semantic_levels
    gamma = cfg.train.gamma
    lam = cfg.train.lam

    # Per-level coefficients c_ℓ (Eq. 9): increasing weight for finer levels
    level_coeffs = [(lvl + 1) / n_levels for lvl in range(n_levels)]

    all_codes_per_level = extra_data.get("all_codes_per_level", {}) if extra_data else {}
    item2sid = extra_data.get("item2sid", {}) if extra_data else {}
    all_items = extra_data.get("all_items", []) if extra_data else []

    policy_params = (
        list(model.hsd.parameters())
        + list(model.ptd.parameters())
        + list(model.mtp_projections.parameters())
    )
    value_params = list(model.hte.parameters())

    policy_optimizer = AdamW(policy_params, lr=cfg.train.hepo_lr_policy)
    value_optimizer = AdamW(value_params, lr=cfg.train.hepo_lr_value)

    best_val_loss = float("inf")

    for epoch in range(cfg.train.hepo_epochs):
        model.train()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_avg_reward = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"HEPO Epoch {epoch+1}/{cfg.train.hepo_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # --- ARR: augment with synthetic anticipatory samples ---
            if cfg.train.arr_enabled and item2sid:
                arr_batch = generate_arr_samples(
                    batch, item2sid, all_items,
                    ratio=cfg.train.arr_synthetic_ratio,
                    n_levels=n_levels,
                )
                batch = merge_batches(batch, arr_batch)

            # --- Simulation: generate K candidates ---
            model.eval()
            with torch.no_grad():
                gen_result = model.generate_candidates(
                    batch, n_candidates=cfg.train.n_candidates
                )
            model.train()

            cand_codes = gen_result["codes"]
            old_logprobs = gen_result["logprobs"]
            pred_values = gen_result["values"]

            B, K, _ = cand_codes.shape
            target_ids = batch["target_ids"]
            target_value = batch["target_value"]

            # --- Popularity-based process rewards (Eq. 6–7) ---
            popularity = _compute_batch_popularity(
                batch["semantic_ids"], batch["token_types"],
                batch["action_types"], n_levels,
            )

            terminal_reward = (
                -torch.abs(pred_values - target_value.unsqueeze(1))
                + (cand_codes[:, :, -1] == target_ids[:, -1].unsqueeze(1)).float()
            )

            rewards = compute_process_rewards(
                cand_codes, popularity, all_codes_per_level,
                terminal_reward, alpha=cfg.train.hepo_alpha,
            )

            # --- Compute per-level cumulative returns G_ℓ (Eq. 10) ---
            returns = torch.zeros_like(rewards)
            for lvl in reversed(range(n_levels)):
                if lvl == n_levels - 1:
                    returns[:, :, lvl] = rewards[:, :, lvl]
                else:
                    returns[:, :, lvl] = rewards[:, :, lvl] + gamma * returns[:, :, lvl + 1]

            # --- Compute advantages with GAE-λ (Eq. 8) ---
            intent = model.hsd(
                batch["semantic_ids"], batch["token_types"],
                batch["user_features"], batch["env_features"],
                batch["seq_len"],
            )
            intent_summary = intent.mean(dim=1).detach()

            advantages = torch.zeros_like(rewards)
            for k_idx in range(K):
                codes_k = cand_codes[:, k_idx, :]

                # V_ϕ(s, z_{1:ℓ-1}) at each level using partial codes
                values_at_levels = []
                for lvl in range(n_levels):
                    partial_codes = codes_k.clone()
                    partial_codes[:, lvl:] = 0
                    _, fv = model.hte(intent_summary, partial_codes)
                    values_at_levels.append(fv.squeeze(-1))
                values_at_levels = torch.stack(values_at_levels, dim=1)  # [B, L]

                # TD errors for intermediate levels
                deltas = []
                for lvl in range(n_levels - 1):
                    delta = (
                        rewards[:, k_idx, lvl]
                        + gamma * values_at_levels[:, lvl + 1]
                        - values_at_levels[:, lvl]
                    )
                    deltas.append(delta)

                # GAE: A_ℓ = Σ_{l=0}^{L-ℓ-1} (γλ)^l · δ_{ℓ+l}
                gae = torch.zeros(B, device=device)
                for lvl in reversed(range(n_levels - 1)):
                    gae = deltas[lvl] + gamma * lam * gae
                    advantages[:, k_idx, lvl] = gae

                # Final level: z-score across candidates (set below)
                advantages[:, k_idx, -1] = rewards[:, k_idx, -1]

            # Z-score normalize final level across K candidates (Eq. 8)
            final_adv = advantages[:, :, -1]
            mu = final_adv.mean(dim=1, keepdim=True)
            sigma = final_adv.std(dim=1, keepdim=True) + 1e-8
            advantages[:, :, -1] = (final_adv - mu) / sigma

            # --- Compute current log-probs ---
            all_new_logprobs = []
            for k_idx in range(K):
                codes_k = cand_codes[:, k_idx, :]
                projected = model.mtp_projections[k_idx % model.n_mtp_heads](intent)
                logits, _ = model.ptd(projected, codes_k)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs.gather(2, codes_k.unsqueeze(-1)).squeeze(-1)
                all_new_logprobs.append(selected)
            new_logprobs = torch.stack(all_new_logprobs, dim=1)  # [B, K, L]

            # --- PPO-clip policy loss with per-level c_ℓ (Eq. 9) ---
            ratio = torch.exp(new_logprobs - old_logprobs.detach())
            eps = cfg.train.clip_eps
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            adv = advantages.detach()

            policy_loss = 0.0
            for lvl in range(n_levels):
                c_l = level_coeffs[lvl]
                lvl_loss = -torch.min(
                    ratio[:, :, lvl] * adv[:, :, lvl],
                    clipped[:, :, lvl] * adv[:, :, lvl],
                ).mean()
                policy_loss = policy_loss + c_l * lvl_loss

            # --- Per-level value loss (Eq. 10) ---
            # L_ϕ = E[ Σ_ℓ (V_ϕ(s, z_{1:ℓ-1}) - G_ℓ)² ]
            value_loss = torch.tensor(0.0, device=device)
            for k_idx in range(K):
                codes_k = cand_codes[:, k_idx, :]
                for lvl in range(n_levels):
                    partial_codes = codes_k.clone()
                    partial_codes[:, lvl:] = 0
                    _, fv = model.hte(intent_summary, partial_codes)
                    value_loss = value_loss + F.mse_loss(
                        fv.squeeze(-1), returns[:, k_idx, lvl].detach()
                    )
            value_loss = value_loss / (K * n_levels)

            total_loss = policy_loss + 0.5 * value_loss

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            policy_optimizer.step()
            value_optimizer.step()

            avg_reward = rewards.sum(dim=-1).mean().item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_avg_reward += avg_reward
            n_batches += 1

            tb.log_scalars("hepo_train_step", {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "total_loss": total_loss.item(),
                "avg_reward": avg_reward,
                "avg_ratio": ratio.mean().item(),
                "grad_norm": grad_norm.item(),
            })
            tb.step()

            pbar.set_postfix(
                policy=f"{policy_loss.item():.4f}",
                value=f"{value_loss.item():.4f}",
                reward=f"{avg_reward:.4f}",
            )

        avg_pl = epoch_policy_loss / max(n_batches, 1)
        avg_vl = epoch_value_loss / max(n_batches, 1)
        avg_rw = epoch_avg_reward / max(n_batches, 1)

        val_loss, val_metrics = evaluate_model(model, val_loader, cfg, mode="mtp")

        tb.log_scalars("hepo_train_epoch", {
            "avg_policy_loss": avg_pl,
            "avg_value_loss": avg_vl,
            "avg_reward": avg_rw,
        }, step=epoch)
        tb.log_scalars("hepo_val_epoch", {
            "loss": val_loss,
            "hitrate_l1": val_metrics["hitrate_l1"],
            "hitrate_full": val_metrics["hitrate_full"],
        }, step=epoch)
        tb.flush()

        print(
            f"  Policy Loss: {avg_pl:.4f} | Value Loss: {avg_vl:.4f} | "
            f"Avg Reward: {avg_rw:.4f} | "
            f"Val HitRate@L1: {val_metrics['hitrate_l1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg.train.save_dir, "hepo_best.pt"))
            print("  -> Saved best HEPO model")

    print(f"HEPO training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, val_loader, cfg, mode="mtp"):
    device = cfg.train.device
    model.eval()

    total_loss = 0.0
    total_l1_hits = 0
    total_full_hits = 0
    total_samples = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        result = model(batch, mode="mtp")

        if mode == "vaft":
            loss, _ = vaft_loss(
                result, batch["target_ids"],
                batch["target_value"], batch["target_action"],
            )
        else:
            loss, _ = mtp_loss(
                result, batch["target_ids"], n_heads=cfg.model.n_mtp_heads,
            )

        B = batch["target_ids"].shape[0]
        total_loss += loss.item() * B

        pred_codes = result["all_logits"][0].argmax(dim=-1)
        target = batch["target_ids"]

        total_l1_hits += (pred_codes[:, 0] == target[:, 0]).sum().item()
        total_full_hits += (pred_codes == target).all(dim=1).sum().item()
        total_samples += B

    avg_loss = total_loss / max(total_samples, 1)
    l1_hitrate = total_l1_hits / max(total_samples, 1)
    full_hitrate = total_full_hits / max(total_samples, 1)

    model.train()
    return avg_loss, {
        "hitrate_l1": l1_hitrate,
        "hitrate_full": full_hitrate,
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(cfg):
    """Load/generate data, train tokenizer, create dataloaders.

    Returns (train_loader, val_loader, extra_data) where extra_data
    contains the semantic trie, popularity maps, and item mappings
    needed by HEPO training.
    """
    print("Preparing data...")

    if cfg.data.dataset == "synthetic":
        print("Using synthetic dataset")
        df, item_meta, item_embeddings, _ = generate_synthetic_data(cfg.data)
    else:
        print(f"Loading Amazon {cfg.data.amazon_category} reviews...")
        try:
            df, item_meta = load_amazon_reviews(cfg.data)
            print(f"  Loaded {len(df)} interactions, {df['item'].nunique()} items")
            n_items = df["item"].nunique()
            item_embeddings = np.random.randn(n_items, cfg.data.item_embed_dim).astype(
                np.float32
            )
            item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        except Exception as e:
            print(f"  Failed to load Amazon data: {e}")
            print("  Falling back to synthetic dataset")
            cfg.data.dataset = "synthetic"
            df, item_meta, item_embeddings, _ = generate_synthetic_data(cfg.data)

    unique_items = sorted(df["item"].unique())
    item2idx = {item: idx for idx, item in enumerate(unique_items)}

    cfg.model.n_items = len(unique_items)
    cfg.model.n_users = df["user"].nunique()
    print(f"  {cfg.model.n_users} users, {cfg.model.n_items} items")

    if len(item_embeddings) < len(unique_items):
        extra = np.random.randn(
            len(unique_items) - len(item_embeddings), cfg.data.item_embed_dim
        ).astype(np.float32)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True) + 1e-8
        item_embeddings = np.vstack([item_embeddings, extra])
    item_embeddings = item_embeddings[:len(unique_items)]

    tokenizer_path = os.path.join(cfg.train.save_dir, "tokenizer.pt")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    tokenizer = RQKMeansPlus(cfg.tokenizer).to(cfg.train.device)

    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training RQ-KMeans+ tokenizer...")
        tokenizer.fit(item_embeddings, cfg.tokenizer)
        tokenizer.save(tokenizer_path)

    all_semantic_ids = tokenizer.encode_all(item_embeddings)
    item2sid = {
        item: all_semantic_ids[item2idx[item]].tolist()
        for item in unique_items
        if item in item2idx
    }

    user_seqs = build_sequences(df, item_meta, cfg.data)
    print(f"  Built {len(user_seqs)} user sequences")

    train_loader, val_loader = create_dataloaders(user_seqs, item2sid, cfg)
    print(
        f"  Train: {len(train_loader.dataset)} samples, "
        f"Val: {len(val_loader.dataset)} samples"
    )

    # Build extra data structures for HEPO
    trie = SemanticTrie.build_from_items(item2sid)
    all_codes_per_level = get_all_codes_per_level(
        item2sid, cfg.model.n_semantic_levels
    )
    user_code_popularity = build_user_code_popularity(
        user_seqs, item2sid, cfg.model.n_semantic_levels
    )

    extra_data = {
        "trie": trie,
        "all_codes_per_level": all_codes_per_level,
        "user_code_popularity": user_code_popularity,
        "item2sid": item2sid,
        "all_items": list(item2sid.keys()),
    }

    return train_loader, val_loader, extra_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPR Training Pipeline")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "amazon"],
                        help="Dataset to use (auto-downloaded if amazon)")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "mtp", "vaft", "hepo"],
                        help="Training stage(s) to run")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mtp_epochs", type=int, default=None)
    parser.add_argument("--vaft_epochs", type=int, default=None)
    parser.add_argument("--hepo_epochs", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Custom run name for TensorBoard")
    args = parser.parse_args()

    # --- Config ---
    cfg = GPRConfig()
    cfg.data.dataset = args.dataset
    if args.device:
        cfg.train.device = args.device
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.mtp_epochs:
        cfg.train.mtp_epochs = args.mtp_epochs
    if args.vaft_epochs:
        cfg.train.vaft_epochs = args.vaft_epochs
    if args.hepo_epochs:
        cfg.train.hepo_epochs = args.hepo_epochs
    cfg = cfg.sync()

    set_seed(cfg.train.seed)

    # --- TensorBoard setup ---
    run_name = args.run_name or (
        f"gpr_{args.dataset}_{args.stage}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    tb_log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb = TBLogger(tb_log_dir)

    config_path = os.path.join(tb_log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "dataset": cfg.data.dataset,
            "stage": args.stage,
            "batch_size": cfg.train.batch_size,
            "d_model": cfg.model.d_model,
            "n_heads": cfg.model.n_heads,
            "n_layers_hsd": cfg.model.n_layers_hsd,
            "n_layers_ptd": cfg.model.n_layers_ptd,
            "n_semantic_levels": cfg.model.n_semantic_levels,
            "codebook_size": cfg.model.codebook_size,
            "n_mtp_heads": cfg.model.n_mtp_heads,
            "mtp_epochs": cfg.train.mtp_epochs,
            "mtp_lr": cfg.train.mtp_lr,
            "vaft_epochs": cfg.train.vaft_epochs,
            "vaft_lr": cfg.train.vaft_lr,
            "hepo_epochs": cfg.train.hepo_epochs,
            "hepo_lr_policy": cfg.train.hepo_lr_policy,
            "hepo_lam": cfg.train.lam,
            "arr_enabled": cfg.train.arr_enabled,
        }, f, indent=2)

    print(f"Device: {cfg.train.device}")
    print(f"Dataset: {cfg.data.dataset}")
    print(f"TensorBoard: {tb_log_dir}")
    print(f"  -> Launch:  tensorboard --logdir {args.log_dir} --port 6006")

    # --- Data ---
    train_loader, val_loader, extra_data = prepare_data(cfg)

    # --- Model ---
    model = GPR(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    tb.log_scalar("model/n_params_M", n_params / 1e6, step=0)
    tb.log_scalar("model/n_layers_hsd", cfg.model.n_layers_hsd, step=0)
    tb.log_scalar("model/d_model", cfg.model.d_model, step=0)

    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, weights_only=True))

    # --- Training ---
    t0 = time.time()

    if args.stage in ("all", "mtp"):
        model = train_mtp(model, train_loader, val_loader, cfg, tb)

    if args.stage in ("all", "vaft"):
        if args.stage == "vaft" and args.resume is None:
            mtp_path = os.path.join(cfg.train.save_dir, "mtp_best.pt")
            if os.path.exists(mtp_path):
                print(f"Loading MTP checkpoint: {mtp_path}")
                model.load_state_dict(torch.load(mtp_path, weights_only=True))
        model = train_vaft(model, train_loader, val_loader, cfg, tb)

    if args.stage in ("all", "hepo"):
        if args.stage == "hepo" and args.resume is None:
            vaft_path = os.path.join(cfg.train.save_dir, "vaft_best.pt")
            if os.path.exists(vaft_path):
                print(f"Loading VAFT checkpoint: {vaft_path}")
                model.load_state_dict(torch.load(vaft_path, weights_only=True))
        model = train_hepo(model, train_loader, val_loader, cfg, tb,
                           extra_data=extra_data)

    elapsed = time.time() - t0
    tb.log_scalar("timing/total_minutes", elapsed / 60, step=0)
    tb.close()

    final_path = os.path.join(cfg.train.save_dir, "gpr_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    print(f"Final model saved to {final_path}")
    print(f"TensorBoard logs at: {tb_log_dir}")
    print(f"  -> Run:  tensorboard --logdir {args.log_dir} --port 6006")


if __name__ == "__main__":
    main()
