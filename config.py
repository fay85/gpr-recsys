from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TokenizerConfig:
    n_levels: int = 3
    codebook_size: int = 256
    embed_dim: int = 64
    input_dim: int = 128
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 512


@dataclass
class DataConfig:
    dataset: str = "amazon"          # "amazon" or "synthetic"
    data_dir: str = "./data"
    amazon_category: str = "Beauty"
    max_seq_len: int = 50
    min_seq_len: int = 5
    n_organic_per_sample: int = 10
    # synthetic data params
    n_users: int = 10000
    n_items: int = 5000
    n_organic_items: int = 3000
    avg_seq_len: int = 20
    item_embed_dim: int = 128


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers_hsd: int = 4
    n_layers_ptd: int = 2
    n_thinking_tokens: int = 4
    n_refining_steps: int = 5
    n_semantic_levels: int = 3
    codebook_size: int = 256
    max_seq_len: int = 50
    n_token_types: int = 4       # U=0, O=1, E=2, I=3
    dropout: float = 0.1
    n_items: int = 5000
    n_users: int = 10000
    d_user: int = 16
    d_env: int = 8
    n_mtp_heads: int = 4
    n_mor_recursions: int = 2    # Mixture-of-Recursions: max extra passes
    n_llm_thought_tokens: int = 4  # external knowledge tokens (Sec. 2.2)
    beam_width: int = 10            # trie beam search width (Sec. 2.3)


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128
    num_workers: int = 4
    dtype: str = "bfloat16"        # "bfloat16" or "float32"

    # Stage 1: MTP pre-training
    mtp_epochs: int = 30
    mtp_lr: float = 1e-3
    mtp_weight_decay: float = 1e-4

    # Stage 2: VAFT
    vaft_epochs: int = 15
    vaft_lr: float = 5e-4

    # Stage 3: HEPO
    hepo_epochs: int = 10
    hepo_lr_policy: float = 1e-4
    hepo_lr_value: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95             # GAE lambda (Eq. 8)
    n_candidates: int = 20
    hepo_alpha: float = 0.1       # process reward scaling

    # Anticipatory Request Rehearsal (Sec. 3.3)
    arr_enabled: bool = True
    arr_synthetic_ratio: float = 0.2

    save_dir: str = "./checkpoints"
    log_interval: int = 50
    eval_interval: int = 1


@dataclass
class GPRConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def sync(self):
        """Ensure consistency between sub-configs."""
        self.model.n_semantic_levels = self.tokenizer.n_levels
        self.model.codebook_size = self.tokenizer.codebook_size
        self.model.max_seq_len = self.data.max_seq_len
        return self
