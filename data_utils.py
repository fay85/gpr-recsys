"""
Data utilities for GPR: Amazon Reviews loader + synthetic data generator.

Amazon Reviews (Beauty 5-core) maps to GPR schema:
  - U-Token: user ID + aggregated stats
  - O-Token: organic content (items from same categories, low engagement)
  - E-Token: request context (time features)
  - I-Token: ad items (items user interacted with)
  - Value:   rating * normalized_price (eCPM proxy)
"""

import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig


# ---------------------------------------------------------------------------
# Amazon Reviews download & preprocessing
# ---------------------------------------------------------------------------

# HuggingFace dataset identifiers (the old UCSD URLs are defunct).
# Ref: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset
AMAZON_HF_DATASETS = {
    "Beauty": "jhan21/amazon-beauty-reviews-dataset",
}

# Local cache directory names (HF Arrow format) that can be placed in the
# project root for offline use.  Created by ``datasets.load_dataset`` or by
# manually downloading from HuggingFace.
AMAZON_LOCAL_DIRS = {
    "Beauty": "jhan21___amazon-beauty-reviews-dataset",
}


def _parse_timestamp(ts_val) -> int:
    """Convert timestamp to unix seconds.  Handles both int (ms or s) and
    ISO-8601 strings like '2020-05-05 14:08:48.923'."""
    if isinstance(ts_val, (int, float)):
        v = int(ts_val)
        return v // 1000 if v > 1e12 else v
    if isinstance(ts_val, str):
        from datetime import datetime
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return int(datetime.strptime(ts_val, fmt).timestamp())
            except ValueError:
                continue
    return 0


def _find_local_arrow(cat: str):
    """Find a local Arrow file for the dataset in the project root or CWD.

    Searches for the HF cache directory structure:
        ``<local_dir>/default/<version>/<hash>/*-train.arrow``
    Returns the path to the Arrow file, or None.
    """
    import glob as _glob
    local_name = AMAZON_LOCAL_DIRS.get(cat)
    if not local_name:
        return None
    project_root = os.path.dirname(os.path.abspath(__file__))
    search_bases = [
        project_root,
        os.path.join(project_root, "dataset"),
        os.getcwd(),
        os.path.join(os.getcwd(), "dataset"),
    ]
    for base in search_bases:
        candidate = os.path.join(base, local_name)
        if not os.path.isdir(candidate):
            continue
        arrows = _glob.glob(os.path.join(candidate, "**", "*-train.arrow"), recursive=True)
        if arrows:
            return arrows[0]
    return None


def load_amazon_reviews(cfg: DataConfig):
    """Load Amazon Beauty reviews, preferring a local copy if present.

    Lookup order:
      1. Local Arrow directory ``<project_root>/jhan21___amazon-beauty-reviews-dataset/``
      2. HuggingFace Hub (requires network; set ``HF_TOKEN`` for auth)

    Dataset: https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset
    Fields:  rating, title, text, asin, parent_asin, user_id, timestamp,
             helpful_vote, verified_purchase
    """
    from datasets import load_dataset, Dataset as HFDataset

    cat = cfg.amazon_category
    if cat not in AMAZON_HF_DATASETS:
        raise ValueError(
            f"Unsupported category: {cat}. "
            f"Available: {list(AMAZON_HF_DATASETS)}"
        )

    # --- Try local Arrow file first (works fully offline) ---
    local_arrow = _find_local_arrow(cat)
    if local_arrow is not None:
        print(f"  Loading local dataset: {local_arrow}")
        ds = HFDataset.from_file(local_arrow)
    else:
        # --- Fall back to HuggingFace Hub ---
        hf_id = AMAZON_HF_DATASETS[cat]
        token = os.environ.get("HF_TOKEN")
        print(f"  Loading HuggingFace dataset: {hf_id}")
        if token:
            print("  Using HF_TOKEN for authenticated access")

        # httpx (used by huggingface_hub) doesn't support SOCKS proxies.
        for var in ("ALL_PROXY", "all_proxy"):
            val = os.environ.get(var, "")
            if val.startswith("socks"):
                print(f"  Unsetting {var}={val} (httpx does not support SOCKS)")
                os.environ.pop(var, None)

        ds = load_dataset(hf_id, split="train", token=token)

    rows = []
    for rec in ds:
        user = rec.get("user_id", "")
        asin = rec.get("asin", "")
        rating = rec.get("rating", 3.0)
        ts = _parse_timestamp(rec.get("timestamp", 0))
        if asin and user and ts:
            rows.append({
                "user": user,
                "item": asin,
                "rating": float(rating),
                "timestamp": ts,
            })

    df = pd.DataFrame(rows).sort_values(["user", "timestamp"]).reset_index(drop=True)

    # Filter 5-core (>= 5 interactions per user and item)
    for _ in range(3):
        item_counts = df["item"].value_counts()
        df = df[df["item"].isin(item_counts[item_counts >= 5].index)]
        user_counts = df["user"].value_counts()
        df = df[df["user"].isin(user_counts[user_counts >= 5].index)]

    # Build lightweight item_meta (the HF dataset has no price/category
    # metadata, so we use defaults; value signal comes from ratings).
    item_meta = {}
    for asin in df["item"].unique():
        item_meta[asin] = {
            "title": asin,
            "categories": [],
            "price": 1.0,
        }

    return df, item_meta


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(cfg: DataConfig):
    """Generate synthetic user-item interaction data matching GPR schema."""
    rng = np.random.RandomState(42)

    item_embeddings = rng.randn(cfg.n_items, cfg.item_embed_dim).astype(np.float32)
    item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

    item_prices = rng.lognormal(mean=2.0, sigma=1.0, size=cfg.n_items).clip(1, 500)

    organic_embeddings = rng.randn(cfg.n_organic_items, cfg.item_embed_dim).astype(np.float32)
    organic_embeddings = organic_embeddings / np.linalg.norm(organic_embeddings, axis=1, keepdims=True)

    rows = []
    for uid in range(cfg.n_users):
        seq_len = rng.poisson(cfg.avg_seq_len) + cfg.min_seq_len
        seq_len = min(seq_len, cfg.max_seq_len * 2)

        n_clusters = rng.randint(1, 4)
        centers = rng.choice(cfg.n_items, size=n_clusters, replace=False)
        center_embs = item_embeddings[centers]

        for t in range(seq_len):
            center = center_embs[rng.randint(0, n_clusters)]
            sims = item_embeddings @ center
            probs = np.exp(sims * 3)
            probs /= probs.sum()
            item_id = rng.choice(cfg.n_items, p=probs)

            sim_val = sims[item_id]
            if sim_val > 0.5:
                action = 2
                rating = rng.choice([4, 5])
            elif sim_val > 0.2:
                action = 1
                rating = rng.choice([3, 4])
            else:
                action = 0
                rating = rng.choice([1, 2, 3])

            rows.append({
                "user": f"u_{uid}",
                "item": f"i_{item_id}",
                "item_idx": item_id,
                "rating": float(rating),
                "timestamp": 1000000 + uid * 10000 + t * 100 + rng.randint(0, 50),
                "action_type": action,
                "price": float(item_prices[item_id]),
            })

    df = pd.DataFrame(rows).sort_values(["user", "timestamp"]).reset_index(drop=True)

    item_meta = {}
    for i in range(cfg.n_items):
        item_meta[f"i_{i}"] = {
            "title": f"item_{i}",
            "categories": [f"cat_{i % 20}", f"subcat_{i % 100}"],
            "price": float(item_prices[i]),
        }

    return df, item_meta, item_embeddings, organic_embeddings


# ---------------------------------------------------------------------------
# Preprocessing: build user sequences and assign token types
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, item_meta: dict, cfg: DataConfig):
    """
    Build user interaction sequences for GPR training.

    Returns list of dicts, each containing:
      - user_id, items, ratings, timestamps, action_types, values
    """
    user_seqs = []
    grouped = df.groupby("user")

    for user, group in grouped:
        group = group.sort_values("timestamp")
        if len(group) < cfg.min_seq_len:
            continue

        items = group["item"].tolist()[-cfg.max_seq_len:]
        ratings = group["rating"].tolist()[-cfg.max_seq_len:]
        timestamps = group["timestamp"].tolist()[-cfg.max_seq_len:]

        action_types = []
        for r in ratings:
            if r >= 4.0:
                action_types.append(2)
            elif r >= 3.0:
                action_types.append(1)
            else:
                action_types.append(0)

        values = []
        for item_id, r in zip(items, ratings):
            price = item_meta.get(item_id, {}).get("price", 1.0)
            values.append(r * np.log1p(price))

        user_seqs.append({
            "user_id": user,
            "items": items,
            "ratings": ratings,
            "timestamps": timestamps,
            "action_types": action_types,
            "values": values,
        })

    return user_seqs


# ---------------------------------------------------------------------------
# User code popularity (for HEPO process rewards, Eq. 6)
# ---------------------------------------------------------------------------

def build_user_code_popularity(user_seqs, item2sid, n_levels):
    """
    Build per-user, per-level code popularity scores P_ℓ(t) ∈ [0,1]
    from successful historical interactions (Eq. 6 in paper).

    Returns dict: user_id → { level → { code → frequency } }
    """
    popularity = {}
    for seq in user_seqs:
        user_id = seq["user_id"]
        items = seq["items"]
        actions = seq["action_types"]

        pop_per_level = {}
        for lvl in range(n_levels):
            code_counts = defaultdict(int)
            total_positive = 0
            for item, action in zip(items, actions):
                if action > 0:  # click or conversion
                    sid = item2sid.get(item)
                    if sid is not None:
                        code_counts[sid[lvl]] += 1
                        total_positive += 1

            if total_positive > 0:
                pop_per_level[lvl] = {
                    c: cnt / total_positive for c, cnt in code_counts.items()
                }
            else:
                pop_per_level[lvl] = {}

        popularity[user_id] = pop_per_level

    return popularity


def get_all_codes_per_level(item2sid, n_levels):
    """Return dict: level → set of all valid codes at that level."""
    result = {lvl: set() for lvl in range(n_levels)}
    for sid in item2sid.values():
        for lvl in range(n_levels):
            result[lvl].add(sid[lvl])
    return result


# ---------------------------------------------------------------------------
# Anticipatory Request Rehearsal (ARR) — Sec. 3.3
# ---------------------------------------------------------------------------

def generate_arr_samples(batch, item2sid, all_items, ratio=0.2, n_levels=3):
    """
    Generate synthetic Anticipatory Request Rehearsal samples.

    Constructs synthetic requests approximating the user's next state:
      - Organic token: refreshed with random recent content
      - User token: reused (user features unchanged)
      - Environment token: time-shifted
    """
    B = batch["semantic_ids"].shape[0]
    n_syn = max(1, int(B * ratio))
    device = batch["semantic_ids"].device

    indices = torch.randint(0, B, (n_syn,))
    synthetic = {k: v[indices].clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # Refresh organic content (O-tokens → random items)
    for i in range(n_syn):
        o_mask = (synthetic["token_types"][i] == 1)
        o_positions = o_mask.nonzero(as_tuple=False).squeeze(-1)
        for pos in o_positions:
            item = random.choice(all_items)
            sid = item2sid.get(item, [0] * n_levels)
            synthetic["semantic_ids"][i, pos] = torch.tensor(sid, dtype=torch.long, device=device)

    # Time-shift environment features
    noise = torch.randn(n_syn, synthetic["env_features"].shape[-1],
                        device=device, dtype=synthetic["env_features"].dtype) * 0.1
    synthetic["env_features"] = synthetic["env_features"] + noise

    return synthetic


def merge_batches(batch_a, batch_b):
    """Concatenate two batches along the batch dimension."""
    merged = {}
    for k in batch_a:
        if isinstance(batch_a[k], torch.Tensor):
            merged[k] = torch.cat([batch_a[k], batch_b[k]], dim=0)
        else:
            merged[k] = batch_a[k]
    return merged


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class GPRDataset(Dataset):
    """
    GPR training dataset.  Each sample contains the unified token
    sequence [U, O..., E, I...] with semantic IDs, user/env features,
    action types, values, and a prediction target.
    """

    def __init__(
        self,
        user_seqs: list,
        item2sid: dict,
        n_levels: int = 3,
        max_seq_len: int = 50,
        n_organic: int = 10,
        d_user: int = 16,
        d_env: int = 8,
        is_train: bool = True,
    ):
        self.n_levels = n_levels
        self.max_seq_len = max_seq_len
        self.n_organic = n_organic
        self.d_user = d_user
        self.d_env = d_env
        self.is_train = is_train
        self.item2sid = item2sid
        self.all_items = list(item2sid.keys())

        self.samples = []
        for seq in user_seqs:
            items = seq["items"]
            if len(items) < 3:
                continue
            n_samples = len(items) - 2 if is_train else 1
            for offset in range(n_samples):
                t = offset + 2 if is_train else len(items) - 1
                self.samples.append({
                    "user_id": seq["user_id"],
                    "history_items": items[:t],
                    "history_actions": seq["action_types"][:t],
                    "history_values": seq["values"][:t],
                    "history_times": seq["timestamps"][:t],
                    "target_item": items[t],
                    "target_action": seq["action_types"][t],
                    "target_value": seq["values"][t],
                })

    def __len__(self):
        return len(self.samples)

    def _get_semantic_id(self, item):
        if item in self.item2sid:
            return self.item2sid[item]
        return [0] * self.n_levels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        hist_items = sample["history_items"]
        hist_actions = sample["history_actions"]
        hist_values = sample["history_values"]

        organic_items = random.sample(
            self.all_items, min(self.n_organic, len(self.all_items))
        )
        max_i_tokens = self.max_seq_len - self.n_organic - 2
        i_items = hist_items[-max_i_tokens:]
        i_actions = hist_actions[-max_i_tokens:]
        i_values = hist_values[-max_i_tokens:]

        seq_len = 1 + len(organic_items) + 1 + len(i_items)
        token_types = (
            [0]
            + [1] * len(organic_items)
            + [2]
            + [3] * len(i_items)
        )

        semantic_ids = []
        semantic_ids.append([0] * self.n_levels)
        for oi in organic_items:
            semantic_ids.append(self._get_semantic_id(oi))
        semantic_ids.append([0] * self.n_levels)
        for ii in i_items:
            semantic_ids.append(self._get_semantic_id(ii))

        user_hash = hash(sample["user_id"]) % (2**31)
        rng = np.random.RandomState(user_hash)
        user_features = rng.randn(self.d_user).astype(np.float32)

        ts = sample["history_times"][-1] if sample["history_times"] else 0
        hour = (ts // 3600) % 24
        dow = (ts // 86400) % 7
        env_features = np.zeros(self.d_env, dtype=np.float32)
        env_features[0] = np.sin(2 * np.pi * hour / 24)
        env_features[1] = np.cos(2 * np.pi * hour / 24)
        env_features[2] = np.sin(2 * np.pi * dow / 7)
        env_features[3] = np.cos(2 * np.pi * dow / 7)

        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            token_types = token_types + [0] * pad_len
            semantic_ids = semantic_ids + [[0] * self.n_levels] * pad_len
        else:
            token_types = token_types[:self.max_seq_len]
            semantic_ids = semantic_ids[:self.max_seq_len]
            seq_len = self.max_seq_len

        padded_actions = i_actions + [0] * (self.max_seq_len - len(i_actions))
        padded_values = i_values + [0.0] * (self.max_seq_len - len(i_values))

        target_ids = self._get_semantic_id(sample["target_item"])

        return {
            "token_types": torch.tensor(token_types, dtype=torch.long),
            "semantic_ids": torch.tensor(semantic_ids, dtype=torch.long),
            "user_features": torch.tensor(user_features, dtype=torch.float32),
            "env_features": torch.tensor(env_features, dtype=torch.float32),
            "action_types": torch.tensor(padded_actions, dtype=torch.long),
            "values": torch.tensor(padded_values, dtype=torch.float32),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_value": torch.tensor(sample["target_value"], dtype=torch.float32),
            "target_action": torch.tensor(sample["target_action"], dtype=torch.long),
        }


def create_dataloaders(
    user_seqs: list,
    item2sid: dict,
    cfg,
    val_ratio: float = 0.1,
):
    """Split sequences and create train/val DataLoaders."""
    random.shuffle(user_seqs)
    split = int(len(user_seqs) * (1 - val_ratio))
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

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.train.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
