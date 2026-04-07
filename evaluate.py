"""
GPR Evaluation & Inference.

Metrics:
  - HitRate@K: fraction of test samples where the target item
    appears in the top-K generated candidates
  - nDCG: normalized Discounted Cumulative Gain for value ranking
  - Code Accuracy: per-level semantic code prediction accuracy

Supports both sampling-based and trie-constrained beam search
(Sec. 2.3) for candidate generation.

Usage:
  python evaluate.py --checkpoint checkpoints/gpr_final.pt
  python evaluate.py --checkpoint checkpoints/gpr_final.pt --dataset amazon
  python evaluate.py --checkpoint checkpoints/gpr_final.pt --use_trie
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import GPRConfig
from data_utils import (
    generate_synthetic_data,
    load_amazon_reviews,
    build_sequences,
    create_dataloaders,
)
from rq_tokenizer import RQKMeansPlus
from model import GPR


@torch.no_grad()
def evaluate(model, val_loader, cfg, n_candidates=20, trie=None):
    """
    Full evaluation with multiple metrics.

    If *trie* is provided, uses Value-Guided Trie-Based Beam Search
    (Sec. 2.3) instead of sampling-based candidate generation.
    """
    device = cfg.train.device
    model = model.to(device)
    model.eval()

    results = {
        "l1_correct": 0,
        "l2_correct": 0,
        "full_correct": 0,
        "total": 0,
        "candidate_hits": 0,
        "ndcg_sum": 0.0,
        "value_mae": 0.0,
    }

    for batch in tqdm(val_loader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        B = batch["target_ids"].shape[0]
        target = batch["target_ids"]

        fwd = model(batch, mode="mtp")
        pred_codes = fwd["all_logits"][0].argmax(dim=-1)

        results["l1_correct"] += (pred_codes[:, 0] == target[:, 0]).sum().item()
        if pred_codes.shape[1] >= 2:
            results["l2_correct"] += (
                (pred_codes[:, :2] == target[:, :2]).all(dim=1).sum().item()
            )
        results["full_correct"] += (pred_codes == target).all(dim=1).sum().item()

        if trie is not None:
            gen = model.trie_beam_search(
                batch, trie,
                beam_width=cfg.model.beam_width,
                n_results=n_candidates,
            )
        else:
            gen = model.generate_candidates(batch, n_candidates=n_candidates)

        cand_codes = gen["codes"]
        cand_values = gen["values"]

        for b in range(B):
            target_code = target[b]
            hit = False
            for k in range(n_candidates):
                if (cand_codes[b, k] == target_code).all():
                    hit = True
                    results["ndcg_sum"] += 1.0 / np.log2(k + 2)
                    break
            if hit:
                results["candidate_hits"] += 1

        if fwd["final_value"] is not None:
            value_pred = fwd["final_value"].squeeze(-1)
            value_gt = batch["target_value"]
            results["value_mae"] += F.l1_loss(
                value_pred, value_gt, reduction="sum"
            ).item()

        results["total"] += B

    T = max(results["total"], 1)

    search_mode = "trie_beam" if trie is not None else "sampling"
    metrics = {
        "Level-1 Accuracy": results["l1_correct"] / T,
        "Level-2 Accuracy": results["l2_correct"] / T,
        "Full Code Accuracy": results["full_correct"] / T,
        f"HitRate@{n_candidates}": results["candidate_hits"] / T,
        "nDCG": results["ndcg_sum"] / T,
        "Value MAE": results["value_mae"] / T,
        "Search Mode": search_mode,
        "Total Samples": results["total"],
    }

    return metrics


def print_metrics(metrics: dict):
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="GPR Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "amazon"])
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_trie", action="store_true",
                        help="Use Value-Guided Trie-Based Beam Search (Sec. 2.3)")
    args = parser.parse_args()

    cfg = GPRConfig()
    cfg.data.dataset = args.dataset
    cfg.train.batch_size = args.batch_size
    if args.device:
        cfg.train.device = args.device
    cfg = cfg.sync()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {cfg.train.device}")

    from train import prepare_data
    _, val_loader, extra_data = prepare_data(cfg)

    model = GPR(cfg.model)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=cfg.train.device, weights_only=True)
    )

    trie = extra_data.get("trie") if args.use_trie else None
    if trie is not None:
        print("Using Value-Guided Trie-Based Beam Search")

    metrics = evaluate(
        model, val_loader, cfg,
        n_candidates=args.n_candidates,
        trie=trie,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
