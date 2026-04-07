# GPR: Generative Pre-trained Recommender

A PyTorch implementation of the GPR model from ["GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation"](https://arxiv.org/abs/2511.10138).

GPR replaces the traditional multi-stage cascading pipeline (retrieval → pre-ranking → ranking) with a single end-to-end generative model for advertising recommendation.

## Architecture

```
User Sequence → [HSD] → Intent Embeddings → [PTD] → Semantic IDs → [HTE] → Values
                  ↑            ↑                ↑                       ↑
          Hybrid Attention   LLM Knowledge   Thinking-Refining-     Per-level
          Token-Aware FFN    Injection        Generation (DDPM)     value heads
          Mixture-of-Recursions
```

### Key Components

| Component | Description |
|-----------|-------------|
| **RQ-KMeans+** | Residual quantizer that maps item embeddings to hierarchical semantic IDs. Combines K-means initialization with VAE-style fine-tuning. Reports collision rate, code usage, and PAS (Path Average Similarity). |
| **HSD** | Heterogeneous Sequence-wise Decoder with hybrid attention (bidirectional for prompt tokens, causal for items), per-token-type FFN/LayerNorm, Mixture-of-Recursions, and external LLM knowledge injection. |
| **PTD** | Progressive Token-wise Decoder with learnable thinking tokens, DDPM-style diffusion refining (cosine noise schedule, ε-prediction), and autoregressive code generation. |
| **HTE** | Hierarchical Token-wise Evaluator that predicts business value at each semantic code level. Uses `stopgrad(h)` to prevent value gradients from updating the backbone. |
| **SemanticTrie** | Trie over valid semantic ID paths for constrained decoding (Sec. 2.3). |

### Inference

Two candidate generation modes are supported:

| Mode | Description |
|------|-------------|
| **Sampling** | Multinomial sampling with temperature and diversity noise across MTP heads. Default. |
| **Trie Beam Search** | Value-Guided Trie-Based Beam Search (Sec. 2.3). Constrains generation to valid item paths and uses HTE value estimates to dynamically guide beam expansion. Enable with `--use_trie`. |

### Training Pipeline

| Stage | Method | Key Details |
|-------|--------|-------------|
| 1 | Multi-Token Prediction (MTP) | N parallel heads predict L-level code paths; uniform simplex-constrained head weights (Eq. 3) |
| 2 | Value-Aware Fine-Tuning (VAFT) | Reweights MTP loss by action type (impression/click/conversion) and normalized eCPM (Eq. 4) |
| 3 | HEPO (RL) | Popularity-based process rewards (Eq. 6–7), GAE-λ advantages (Eq. 8), per-level PPO-clip coefficients c_ℓ (Eq. 9), per-level value loss (Eq. 10), and Anticipatory Request Rehearsal (ARR) |

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Train with synthetic data (no download needed)

```bash
python train.py --dataset synthetic --batch_size 128
```

### Train with Amazon Reviews

```bash
python train.py --dataset amazon --batch_size 128
```

The Amazon Beauty dataset will be downloaded automatically.

### Run individual stages

```bash
# Stage 1: MTP pre-training
python train.py --stage mtp --dataset synthetic

# Stage 2: VAFT (loads MTP checkpoint automatically)
python train.py --stage vaft

# Stage 3: HEPO (loads VAFT checkpoint automatically)
python train.py --stage hepo
```

### Evaluate

```bash
# Sampling-based candidate generation
python evaluate.py --checkpoint checkpoints/gpr_final.pt --dataset synthetic

# Trie-constrained value-guided beam search (Sec. 2.3)
python evaluate.py --checkpoint checkpoints/gpr_final.pt --use_trie
```

### Monitor training

```bash
tensorboard --logdir runs/ --port 6006
```

## Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Model
d_model = 128              # hidden dimension
n_heads = 4                # attention heads
n_layers_hsd = 4           # HSD transformer layers
n_layers_ptd = 2           # PTD decoder layers
n_semantic_levels = 3      # hierarchy depth
codebook_size = 256        # codes per level
n_mtp_heads = 4            # multi-token prediction heads
n_mor_recursions = 2       # Mixture-of-Recursions depth
n_llm_thought_tokens = 4   # external knowledge tokens
n_refining_steps = 5       # DDPM diffusion steps
beam_width = 10            # trie beam search width

# Training
mtp_epochs = 30            # Stage 1
vaft_epochs = 15           # Stage 2
hepo_epochs = 10           # Stage 3
batch_size = 128
gamma = 0.99               # discount factor
lam = 0.95                 # GAE lambda
arr_enabled = True         # Anticipatory Request Rehearsal
arr_synthetic_ratio = 0.2  # fraction of synthetic ARR samples
```

## Data Schema

GPR uses four token types to represent the user journey:

| Token | Content | Example |
|-------|---------|---------|
| **U-Token** | User attributes | Demographics, preferences |
| **O-Token** | Organic content | Videos, articles browsed |
| **E-Token** | Environment/context | Time, device, position |
| **I-Token** | Ad items | Ads user interacted with |

Items are encoded as L-level semantic IDs via RQ-KMeans+ quantization.

## Project Structure

```
gpr-recsys/
├── config.py               # Dataclass configurations
├── data_utils.py           # Data loading, popularity tracking, ARR
├── rq_tokenizer.py         # RQ-KMeans+ tokenizer (with PAS metric)
├── model.py                # GPR model (HSD, PTD, HTE, SemanticTrie)
├── train.py                # 3-stage training pipeline
├── evaluate.py             # Evaluation (sampling + trie beam search)
├── plot_runs_comparison.py # TensorBoard log visualization
├── requirements.txt
└── README.md
```

## Paper Correspondence

| Paper Section | Implementation |
|---------------|----------------|
| Sec. 2.1 — Input Schema & RQ-KMeans+ | `data_utils.py`, `rq_tokenizer.py` |
| Sec. 2.2 — HSD (Hybrid Attn, Token-Aware FFN/Norm, MoR) | `model.py: HSD, HSDBlock, MixtureOfRecursions` |
| Sec. 2.2 — External LLM Knowledge | `model.py: LLMKnowledgeModule` |
| Sec. 2.2 — PTD (Thinking-Refining-Generation) | `model.py: PTD, RefiningModule` |
| Sec. 2.2 — HTE (value estimation + critic) | `model.py: HTE` |
| Sec. 2.3 — Value-Guided Trie-Based Beam Search | `model.py: SemanticTrie, GPR.trie_beam_search` |
| Sec. 3.1 — MTP Pre-training (Eq. 3) | `train.py: train_mtp`, `model.py: mtp_loss` |
| Sec. 3.2 — VAFT (Eq. 4) | `train.py: train_vaft`, `model.py: vaft_loss` |
| Sec. 3.3 — HEPO: Process Rewards (Eq. 6–7) | `train.py: compute_process_rewards` |
| Sec. 3.3 — HEPO: GAE-λ Advantages (Eq. 8) | `train.py: train_hepo` (GAE loop) |
| Sec. 3.3 — HEPO: PPO-clip with c_ℓ (Eq. 9) | `train.py: train_hepo` (level_coeffs) |
| Sec. 3.3 — HEPO: Per-level Value Loss (Eq. 10) | `train.py: train_hepo` (value_loss loop) |
| Sec. 3.3 — Anticipatory Request Rehearsal | `data_utils.py: generate_arr_samples` |
| Table 1 — Tokenizer Metrics | `rq_tokenizer.py: _compute_metrics` (Collision, CUR, PAS) |

## Citation

```bibtex
@article{zhang2025gpr,
  title={GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation},
  author={Zhang, Jun and Li, Yi and Liu, Yue and Wang, Changping and others},
  journal={arXiv preprint arXiv:2511.10138},
  year={2025}
}
```
