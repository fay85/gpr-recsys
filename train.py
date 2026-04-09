"""
GPR Training Pipeline with FSDP + TensorBoard Monitoring.

Three-stage training:
  Stage 1: Pre-training with Multi-Token Prediction (MTP)
  Stage 2: Value-Aware Fine-Tuning (VAFT)
  Stage 3: Post-training with HEPO (Hierarchical Enhanced Policy Optimization)

Launch with torchrun for multi-GPU FSDP:
  torchrun --nproc_per_node=8 train.py --dataset amazon

Single-GPU fallback (no FSDP):
  python train.py --dataset amazon --no_fsdp

All losses, metrics, and learning rates are logged to TensorBoard.
"""

import os
import argparse
import functools
import random
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
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
from model import GPR, SemanticTrie, mtp_loss, vaft_loss, HSDBlock


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_dist():
    return dist.is_initialized()


def rank():
    return dist.get_rank() if is_dist() else 0


def world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main():
    return rank() == 0


def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def print0(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def setup_distributed():
    if "RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank())
    return True


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_model_fsdp(model, cfg):
    """Wrap GPR model with FSDP using per-HSDBlock sharding."""
    device_id = local_rank()
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            HSDBlock,
            nn.TransformerDecoderLayer,
        },
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    if cfg.train.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: isinstance(m, (HSDBlock,)),
        )

    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers for FSDP
# ---------------------------------------------------------------------------

def save_checkpoint(model, path, is_fsdp=False):
    if is_fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()
        if is_main():
            torch.save(state, path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path, is_fsdp=False):
    if is_fsdp:
        full_sd = None
        if is_main():
            full_sd = torch.load(path, weights_only=True, map_location="cpu")
        if is_dist():
            objects = [full_sd]
            dist.broadcast_object_list(objects, src=0)
            full_sd = objects[0]
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model.load_state_dict(full_sd)
    else:
        model.load_state_dict(torch.load(path, weights_only=True))


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def to_device(batch: dict, device, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


def clip_grad_norm(model, max_norm, is_fsdp=False):
    if is_fsdp:
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ---------------------------------------------------------------------------
# TensorBoard helper (rank-0 only)
# ---------------------------------------------------------------------------

class TBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir) if is_main() else None
        self.global_step = 0

    def log_scalars(self, tag_prefix: str, scalars: dict, step: int = None):
        if self.writer is None:
            return
        s = step if step is not None else self.global_step
        for k, v in scalars.items():
            self.writer.add_scalar(f"{tag_prefix}/{k}", v, s)

    def log_scalar(self, tag: str, value: float, step: int = None):
        if self.writer is None:
            return
        s = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, s)

    def step(self):
        self.global_step += 1

    def flush(self):
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()


# ---------------------------------------------------------------------------
# Stage 1: Pre-training with Multi-Token Prediction
# ---------------------------------------------------------------------------

def train_mtp(model, train_loader, val_loader, cfg, tb: TBLogger,
              is_fsdp=False):
    print0("\n" + "=" * 60)
    print0("Stage 1: Pre-training with Multi-Token Prediction (MTP)")
    print0("=" * 60)

    device = torch.device("cuda", local_rank())
    dtype = get_dtype(cfg.train.dtype)

    if not is_fsdp:
        model = model.to(device=device, dtype=dtype)

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
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_metrics = {"ce_loss": 0.0, "refine_loss": 0.0}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"MTP Epoch {epoch+1}/{cfg.train.mtp_epochs}",
            disable=not is_main(),
        )
        for batch in pbar:
            batch = to_device(batch, device, dtype)

            result = model(batch, mode="mtp")
            loss, metrics = mtp_loss(
                result, batch["target_ids"], n_heads=cfg.model.n_mtp_heads
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, 1.0, is_fsdp)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            tb.log_scalars("mtp_train_step", {
                "total_loss": loss.item(),
                "ce_loss": metrics["ce_loss"],
                "refine_loss": metrics["refine_loss"],
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            })
            tb.step()

            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        val_loss, val_metrics = evaluate_model(
            model, val_loader, cfg, mode="mtp", is_fsdp=is_fsdp
        )

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

        print0(
            f"  Train Loss: {avg_loss:.4f} | "
            f"CE: {avg_metrics['ce_loss']:.4f} | "
            f"Refine: {avg_metrics['refine_loss']:.4f}"
        )
        print0(
            f"  Val Loss:   {val_loss:.4f} | "
            f"HitRate@L1: {val_metrics['hitrate_l1']:.4f} | "
            f"HitRate@Full: {val_metrics['hitrate_full']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                os.path.join(cfg.train.save_dir, "mtp_best.pt"),
                is_fsdp=is_fsdp,
            )
            print0("  -> Saved best MTP model")

    print0(f"MTP training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 2: Value-Aware Fine-Tuning (VAFT)
# ---------------------------------------------------------------------------

def train_vaft(model, train_loader, val_loader, cfg, tb: TBLogger,
               is_fsdp=False):
    print0("\n" + "=" * 60)
    print0("Stage 2: Value-Aware Fine-Tuning (VAFT)")
    print0("=" * 60)

    device = torch.device("cuda", local_rank())
    dtype = get_dtype(cfg.train.dtype)

    if not is_fsdp:
        model = model.to(device=device, dtype=dtype)

    optimizer = AdamW(model.parameters(), lr=cfg.train.vaft_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.vaft_epochs, eta_min=cfg.train.vaft_lr * 0.01
    )

    best_val_loss = float("inf")

    for epoch in range(cfg.train.vaft_epochs):
        model.train()
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_metrics = {"ce_loss": 0.0, "value_loss": 0.0}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"VAFT Epoch {epoch+1}/{cfg.train.vaft_epochs}",
            disable=not is_main(),
        )
        for batch in pbar:
            batch = to_device(batch, device, dtype)

            result = model(batch, mode="vaft")
            loss, metrics = vaft_loss(
                result, batch["target_ids"],
                batch["target_value"], batch["target_action"],
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, 1.0, is_fsdp)
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

            tb.log_scalars("vaft_train_step", {
                "total_loss": loss.item(),
                "ce_loss": metrics["ce_loss"],
                "value_loss": metrics["value_loss"],
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
            })
            tb.step()

            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        val_loss, val_metrics = evaluate_model(
            model, val_loader, cfg, mode="vaft", is_fsdp=is_fsdp
        )

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

        print0(
            f"  Train Loss: {avg_loss:.4f} | "
            f"CE: {avg_metrics['ce_loss']:.4f} | "
            f"Value: {avg_metrics['value_loss']:.4f}"
        )
        print0(
            f"  Val Loss:   {val_loss:.4f} | "
            f"HitRate@L1: {val_metrics['hitrate_l1']:.4f} | "
            f"HitRate@Full: {val_metrics['hitrate_full']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                os.path.join(cfg.train.save_dir, "vaft_best.pt"),
                is_fsdp=is_fsdp,
            )
            print0("  -> Saved best VAFT model")

    print0(f"VAFT training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Stage 3: Post-training with HEPO
# ---------------------------------------------------------------------------

def _compute_batch_popularity(semantic_ids, token_types, action_types, n_levels):
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
            total = counts.sum().item()
            pop_per_level[lvl] = {
                c.item(): cnt.item() / total
                for c, cnt in zip(unique_codes, counts)
            }

        popularity.append(pop_per_level)

    return popularity


def compute_process_rewards(codes, popularity, all_codes_per_level,
                            terminal_rewards, alpha=0.1):
    B, K, n_levels = codes.shape
    device = codes.device
    rewards = torch.zeros(B, K, n_levels, device=device, dtype=terminal_rewards.dtype)

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
               extra_data=None, is_fsdp=False):
    print0("\n" + "=" * 60)
    print0("Stage 3: Post-training with HEPO")
    print0("=" * 60)

    device = torch.device("cuda", local_rank())
    dtype = get_dtype(cfg.train.dtype)

    if not is_fsdp:
        model = model.to(device=device, dtype=dtype)

    n_levels = cfg.model.n_semantic_levels
    gamma = cfg.train.gamma
    lam = cfg.train.lam

    level_coeffs = [(lvl + 1) / n_levels for lvl in range(n_levels)]

    all_codes_per_level = extra_data.get("all_codes_per_level", {}) if extra_data else {}
    item2sid = extra_data.get("item2sid", {}) if extra_data else {}
    all_items = extra_data.get("all_items", []) if extra_data else []

    policy_params = []
    value_params = []
    for name, param in model.named_parameters():
        if "hte" in name:
            value_params.append(param)
        else:
            policy_params.append(param)

    policy_optimizer = AdamW(policy_params, lr=cfg.train.hepo_lr_policy)
    value_optimizer = AdamW(value_params, lr=cfg.train.hepo_lr_value)

    best_val_loss = float("inf")

    for epoch in range(cfg.train.hepo_epochs):
        model.train()
        if hasattr(train_loader, "sampler") and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_avg_reward = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"HEPO Epoch {epoch+1}/{cfg.train.hepo_epochs}",
            disable=not is_main(),
        )
        for batch in pbar:
            batch = to_device(batch, device, dtype)

            if cfg.train.arr_enabled and item2sid:
                arr_batch = generate_arr_samples(
                    batch, item2sid, all_items,
                    ratio=cfg.train.arr_synthetic_ratio,
                    n_levels=n_levels,
                )
                batch = merge_batches(batch, arr_batch)

            model.eval()
            with torch.no_grad():
                gen_result = model(batch, mode="hepo_generate")
                gen_result_full = model.module.generate_candidates(
                    batch, n_candidates=cfg.train.n_candidates
                ) if is_fsdp else model.generate_candidates(
                    batch, n_candidates=cfg.train.n_candidates
                )
            model.train()

            cand_codes = gen_result_full["codes"]
            old_logprobs = gen_result_full["logprobs"]
            pred_values = gen_result_full["values"]

            B, K, _ = cand_codes.shape
            target_ids = batch["target_ids"]
            target_value = batch["target_value"]

            popularity = _compute_batch_popularity(
                batch["semantic_ids"], batch["token_types"],
                batch["action_types"], n_levels,
            )

            terminal_reward = (
                -torch.abs(pred_values - target_value.unsqueeze(1))
                + (cand_codes[:, :, -1] == target_ids[:, -1].unsqueeze(1)).to(dtype)
            )

            rewards = compute_process_rewards(
                cand_codes, popularity, all_codes_per_level,
                terminal_reward, alpha=cfg.train.hepo_alpha,
            )

            returns = torch.zeros_like(rewards)
            for lvl in reversed(range(n_levels)):
                if lvl == n_levels - 1:
                    returns[:, :, lvl] = rewards[:, :, lvl]
                else:
                    returns[:, :, lvl] = rewards[:, :, lvl] + gamma * returns[:, :, lvl + 1]

            _model = model.module if is_fsdp else model
            intent = _model.hsd(
                batch["semantic_ids"], batch["token_types"],
                batch["user_features"], batch["env_features"],
                batch["seq_len"],
            )
            intent_summary = intent.mean(dim=1).detach()

            advantages = torch.zeros_like(rewards)
            for k_idx in range(K):
                codes_k = cand_codes[:, k_idx, :]

                values_at_levels = []
                for lvl in range(n_levels):
                    partial_codes = codes_k.clone()
                    partial_codes[:, lvl:] = 0
                    _, fv = _model.hte(intent_summary, partial_codes)
                    values_at_levels.append(fv.squeeze(-1))
                values_at_levels = torch.stack(values_at_levels, dim=1)

                deltas = []
                for lvl in range(n_levels - 1):
                    delta = (
                        rewards[:, k_idx, lvl]
                        + gamma * values_at_levels[:, lvl + 1]
                        - values_at_levels[:, lvl]
                    )
                    deltas.append(delta)

                gae = torch.zeros(B, device=device, dtype=dtype)
                for lvl in reversed(range(n_levels - 1)):
                    gae = deltas[lvl] + gamma * lam * gae
                    advantages[:, k_idx, lvl] = gae

                advantages[:, k_idx, -1] = rewards[:, k_idx, -1]

            final_adv = advantages[:, :, -1]
            mu = final_adv.mean(dim=1, keepdim=True)
            sigma = final_adv.std(dim=1, keepdim=True) + 1e-8
            advantages[:, :, -1] = (final_adv - mu) / sigma

            all_new_logprobs = []
            for k_idx in range(K):
                codes_k = cand_codes[:, k_idx, :]
                projected = _model.mtp_projections[k_idx % _model.n_mtp_heads](intent)
                logits, _ = _model.ptd(projected, codes_k)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs.gather(2, codes_k.unsqueeze(-1)).squeeze(-1)
                all_new_logprobs.append(selected)
            new_logprobs = torch.stack(all_new_logprobs, dim=1)

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

            value_loss = torch.tensor(0.0, device=device, dtype=dtype)
            for k_idx in range(K):
                codes_k = cand_codes[:, k_idx, :]
                for lvl in range(n_levels):
                    partial_codes = codes_k.clone()
                    partial_codes[:, lvl:] = 0
                    _, fv = _model.hte(intent_summary, partial_codes)
                    value_loss = value_loss + F.mse_loss(
                        fv.squeeze(-1), returns[:, k_idx, lvl].detach()
                    )
            value_loss = value_loss / (K * n_levels)

            total_loss = policy_loss + 0.5 * value_loss

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = clip_grad_norm(model, 1.0, is_fsdp)
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
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            })
            tb.step()

            if is_main():
                pbar.set_postfix(
                    policy=f"{policy_loss.item():.4f}",
                    value=f"{value_loss.item():.4f}",
                    reward=f"{avg_reward:.4f}",
                )

        avg_pl = epoch_policy_loss / max(n_batches, 1)
        avg_vl = epoch_value_loss / max(n_batches, 1)
        avg_rw = epoch_avg_reward / max(n_batches, 1)

        val_loss, val_metrics = evaluate_model(
            model, val_loader, cfg, mode="mtp", is_fsdp=is_fsdp
        )

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

        print0(
            f"  Policy Loss: {avg_pl:.4f} | Value Loss: {avg_vl:.4f} | "
            f"Avg Reward: {avg_rw:.4f} | "
            f"Val HitRate@L1: {val_metrics['hitrate_l1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                os.path.join(cfg.train.save_dir, "hepo_best.pt"),
                is_fsdp=is_fsdp,
            )
            print0("  -> Saved best HEPO model")

    print0(f"HEPO training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, val_loader, cfg, mode="mtp", is_fsdp=False):
    device = torch.device("cuda", local_rank())
    dtype = get_dtype(cfg.train.dtype)
    model.eval()

    total_loss = 0.0
    total_l1_hits = 0
    total_full_hits = 0
    total_samples = 0

    for batch in val_loader:
        batch = to_device(batch, device, dtype)

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

    if is_dist():
        stats = torch.tensor(
            [total_loss, total_l1_hits, total_full_hits, total_samples],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_l1_hits, total_full_hits, total_samples = stats.tolist()

    avg_loss = total_loss / max(total_samples, 1)
    l1_hitrate = total_l1_hits / max(total_samples, 1)
    full_hitrate = total_full_hits / max(total_samples, 1)

    model.train()
    return avg_loss, {
        "hitrate_l1": l1_hitrate,
        "hitrate_full": full_hitrate,
    }


# ---------------------------------------------------------------------------
# Data preparation (runs on all ranks, but downloads on rank 0 only)
# ---------------------------------------------------------------------------

def prepare_data(cfg, is_distributed=False):
    print0("Preparing data...")

    if cfg.data.dataset == "synthetic":
        print0("Using synthetic dataset")
        df, item_meta, item_embeddings, _ = generate_synthetic_data(cfg.data)
    else:
        print0(f"Loading Amazon {cfg.data.amazon_category} reviews...")
        try:
            df, item_meta = load_amazon_reviews(cfg.data)
            print0(f"  Loaded {len(df)} interactions, {df['item'].nunique()} items")
            n_items = df["item"].nunique()
            item_embeddings = np.random.randn(n_items, cfg.data.item_embed_dim).astype(
                np.float32
            )
            item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        except Exception as e:
            print0(f"  Failed to load Amazon data: {e}")
            print0("  Falling back to synthetic dataset")
            cfg.data.dataset = "synthetic"
            df, item_meta, item_embeddings, _ = generate_synthetic_data(cfg.data)

    unique_items = sorted(df["item"].unique())
    item2idx = {item: idx for idx, item in enumerate(unique_items)}

    cfg.model.n_items = len(unique_items)
    cfg.model.n_users = df["user"].nunique()
    print0(f"  {cfg.model.n_users} users, {cfg.model.n_items} items")

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
        print0("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print0("Training RQ-KMeans+ tokenizer...")
        tokenizer.fit(item_embeddings, cfg.tokenizer)
        if is_main():
            tokenizer.save(tokenizer_path)
        if is_distributed:
            dist.barrier()

    all_semantic_ids = tokenizer.encode_all(item_embeddings)
    item2sid = {
        item: all_semantic_ids[item2idx[item]].tolist()
        for item in unique_items
        if item in item2idx
    }

    user_seqs = build_sequences(df, item_meta, cfg.data)
    print0(f"  Built {len(user_seqs)} user sequences")

    train_loader, val_loader = create_dataloaders_distributed(
        user_seqs, item2sid, cfg, is_distributed
    )
    print0(
        f"  Train: {len(train_loader.dataset)} samples, "
        f"Val: {len(val_loader.dataset)} samples"
    )

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


def create_dataloaders_distributed(user_seqs, item2sid, cfg, is_distributed):
    """Create DataLoaders with DistributedSampler when running multi-GPU."""
    from data_utils import GPRDataset

    random.shuffle(user_seqs)
    split = int(len(user_seqs) * 0.9)
    train_seqs = user_seqs[:split]
    val_seqs = user_seqs[split:]

    train_ds = GPRDataset(
        train_seqs, item2sid,
        n_levels=cfg.model.n_semantic_levels,
        max_seq_len=cfg.data.max_seq_len,
        n_organic=cfg.data.n_organic_per_sample,
        d_user=cfg.model.d_user,
        d_env=cfg.model.d_env,
        is_train=True,
    )
    val_ds = GPRDataset(
        val_seqs, item2sid,
        n_levels=cfg.model.n_semantic_levels,
        max_seq_len=cfg.data.max_seq_len,
        n_organic=cfg.data.n_organic_per_sample,
        d_user=cfg.model.d_user,
        d_env=cfg.model.d_env,
        is_train=False,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPR Training Pipeline (FSDP)")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "amazon"])
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "mtp", "vaft", "hepo"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mtp_epochs", type=int, default=None)
    parser.add_argument("--vaft_epochs", type=int, default=None)
    parser.add_argument("--hepo_epochs", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--no_fsdp", action="store_true",
                        help="Disable FSDP (single-GPU mode)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    is_distributed = setup_distributed()
    use_fsdp = is_distributed and not args.no_fsdp

    cfg = GPRConfig()
    cfg.data.dataset = args.dataset
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.mtp_epochs:
        cfg.train.mtp_epochs = args.mtp_epochs
    if args.vaft_epochs:
        cfg.train.vaft_epochs = args.vaft_epochs
    if args.hepo_epochs:
        cfg.train.hepo_epochs = args.hepo_epochs
    cfg = cfg.sync()

    set_seed(cfg.train.seed + rank())

    run_name = args.run_name or (
        f"gpr_fsdp{world_size()}_{args.dataset}_{args.stage}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    tb_log_dir = os.path.join(args.log_dir, run_name)
    if is_main():
        os.makedirs(tb_log_dir, exist_ok=True)
    tb = TBLogger(tb_log_dir)

    if is_main():
        config_path = os.path.join(tb_log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
                "dataset": cfg.data.dataset,
                "stage": args.stage,
                "world_size": world_size(),
                "use_fsdp": use_fsdp,
                "batch_size_per_gpu": cfg.train.batch_size,
                "effective_batch_size": cfg.train.batch_size * world_size(),
                "d_model": cfg.model.d_model,
                "n_heads": cfg.model.n_heads,
                "d_ff": cfg.model.d_ff,
                "n_layers_hsd": cfg.model.n_layers_hsd,
                "n_layers_ptd": cfg.model.n_layers_ptd,
                "n_semantic_levels": cfg.model.n_semantic_levels,
                "codebook_size": cfg.model.codebook_size,
                "max_seq_len": cfg.model.max_seq_len,
                "n_mtp_heads": cfg.model.n_mtp_heads,
                "mtp_epochs": cfg.train.mtp_epochs,
                "mtp_lr": cfg.train.mtp_lr,
                "vaft_epochs": cfg.train.vaft_epochs,
                "vaft_lr": cfg.train.vaft_lr,
                "hepo_epochs": cfg.train.hepo_epochs,
                "activation_checkpointing": cfg.train.activation_checkpointing,
            }, f, indent=2)

    print0(f"World size: {world_size()}")
    print0(f"FSDP: {use_fsdp}")
    print0(f"Device: cuda:{local_rank()}")
    print0(f"Dtype: {cfg.train.dtype}")
    print0(f"Model: d={cfg.model.d_model}, HSD layers={cfg.model.n_layers_hsd}, "
           f"PTD layers={cfg.model.n_layers_ptd}")
    print0(f"Batch size per GPU: {cfg.train.batch_size}, "
           f"Effective: {cfg.train.batch_size * world_size()}")
    print0(f"Dataset: {cfg.data.dataset}")
    print0(f"TensorBoard: {tb_log_dir}")

    train_loader, val_loader, extra_data = prepare_data(cfg, is_distributed)

    model = GPR(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print0(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    tb.log_scalar("model/n_params_M", n_params / 1e6, step=0)
    tb.log_scalar("model/n_layers_hsd", cfg.model.n_layers_hsd, step=0)
    tb.log_scalar("model/d_model", cfg.model.d_model, step=0)

    if use_fsdp:
        model = wrap_model_fsdp(model, cfg)
        print0("Model wrapped with FSDP")
        if cfg.train.activation_checkpointing:
            print0("Activation checkpointing enabled for HSDBlock")
    else:
        device = torch.device("cuda", local_rank())
        dtype = get_dtype(cfg.train.dtype)
        model = model.to(device=device, dtype=dtype)

    if args.resume:
        print0(f"Loading checkpoint: {args.resume}")
        load_checkpoint(model, args.resume, is_fsdp=use_fsdp)

    t0 = time.time()

    if args.stage in ("all", "mtp"):
        model = train_mtp(model, train_loader, val_loader, cfg, tb,
                          is_fsdp=use_fsdp)

    if args.stage in ("all", "vaft"):
        if args.stage == "vaft" and args.resume is None:
            mtp_path = os.path.join(cfg.train.save_dir, "mtp_best.pt")
            if os.path.exists(mtp_path):
                print0(f"Loading MTP checkpoint: {mtp_path}")
                load_checkpoint(model, mtp_path, is_fsdp=use_fsdp)
        model = train_vaft(model, train_loader, val_loader, cfg, tb,
                           is_fsdp=use_fsdp)

    if args.stage in ("all", "hepo"):
        if args.stage == "hepo" and args.resume is None:
            vaft_path = os.path.join(cfg.train.save_dir, "vaft_best.pt")
            if os.path.exists(vaft_path):
                print0(f"Loading VAFT checkpoint: {vaft_path}")
                load_checkpoint(model, vaft_path, is_fsdp=use_fsdp)
        model = train_hepo(model, train_loader, val_loader, cfg, tb,
                           extra_data=extra_data, is_fsdp=use_fsdp)

    elapsed = time.time() - t0
    tb.log_scalar("timing/total_minutes", elapsed / 60, step=0)
    tb.close()

    final_path = os.path.join(cfg.train.save_dir, "gpr_final.pt")
    save_checkpoint(model, final_path, is_fsdp=use_fsdp)
    print0(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    print0(f"Final model saved to {final_path}")
    print0(f"TensorBoard logs at: {tb_log_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
