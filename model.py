"""
GPR Model: Generative Pre-trained Recommender.

Architecture (Heterogeneous Hierarchical Decoder — HHD):
  1. HSD (Heterogeneous Sequence-wise Decoder)
     - Stacks HSTU blocks [Zhai et al. 2024] as the foundational layer
     - Enhancements on top of HSTU:
       * Hybrid Attention mask (bidirectional for U/O/E prompt, causal for I)
       * Token-Aware LayerNorm (separate per token type)
       * Token-Aware FFN (separate per token type)
       * Mixture-of-Recursions (per-token adaptive depth)
       * External LLM knowledge injection (Sec. 2.2)
     → outputs intent embeddings

  2. PTD (Progressive Token-wise Decoder)
     - Thinking tokens: learnable queries cross-attending to intent
     - Refining module: DDPM-style diffusion denoiser with cosine schedule
     - Generation: autoregressive semantic code prediction
     → outputs L-level semantic ID codes

  3. HTE (Hierarchical Token-wise Evaluator)
     - Per-level value heads
     - Final aggregated value prediction
     → outputs value estimates for beam search & RL

Reference for HSTU block:
  "Actions Speak Louder than Words: Trillion-Parameter Sequential
   Transducers for Generative Recommendations" (Zhai et al., 2024)
  Key: Attention(X) = softmax(QK^T/√d + M) V ⊙ U
  where U is a learned pointwise modulation (no FFN layer in vanilla HSTU).
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# =========================================================================
# HSTU Core Block (faithful to Zhai et al. 2024)
# =========================================================================

class HSTUAttention(nn.Module):
    """
    HSTU-style multi-head attention with pointwise modulation.

    From the HSTU paper: the output is
        Attention(X) = softmax(QK^T / √d) · V  ⊙  U
    where Q, K, V, U are all linear projections of X (bias-free),
    and ⊙ denotes element-wise product.  U replaces the FFN in
    vanilla HSTU — it provides per-element non-linear gating.

    GPR extends this with:
      1. Hybrid attention mask  M_hybrid  added inside the softmax
      2. The mask is bidirectional among prompt tokens (U/O/E-Token)
         and causal among item tokens (I-Token)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # HSTU uses bias-free projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_u = nn.Linear(d_model, d_model, bias=False)  # pointwise modulation
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_types, seq_lens=None):
        """
        x: [B, L, D]
        token_types: [B, L] with 0=U, 1=O, 2=E, 3=I
        Returns: [B, L, D]
        """
        B, L, D = x.shape
        H, dk = self.n_heads, self.d_k

        Q = self.W_q(x).view(B, L, H, dk).transpose(1, 2)  # [B, H, L, dk]
        K = self.W_k(x).view(B, L, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, dk).transpose(1, 2)
        U = self.W_u(x).view(B, L, H, dk).transpose(1, 2)  # modulation

        # Scaled dot-product + hybrid mask
        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)    # [B, H, L, L]

        # Hybrid attention mask (Eq. 2 in GPR paper)
        mask = self._build_hybrid_mask(token_types, L, x.device, x.dtype)
        scores = scores + mask.unsqueeze(1)                        # broadcast over heads

        # Padding mask
        if seq_lens is not None:
            pad_mask = torch.arange(L, device=x.device).unsqueeze(0) >= seq_lens.unsqueeze(1)
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2).expand(-1, H, L, -1),
                torch.finfo(scores.dtype).min,
            )

        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # HSTU core: (A · V) ⊙ U   — element-wise modulation, NO sigmoid
        out = (attn_weights @ V) * U                         # [B, H, L, dk]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)

    @staticmethod
    def _build_hybrid_mask(token_types, L, device, dtype):
        """
        Build M_hybrid (Eq. 2 in GPR paper):
          M[i,j] = 0       if i >= j  OR  both X_i, X_j ∈ {U/O/E-Token}
          M[i,j] = -inf    if j > i  (causal among I-tokens)
        Prompt tokens use bidirectional attention; I-tokens see all
        prompt tokens plus earlier I-tokens (causal).
        """
        B = token_types.shape[0]
        is_prompt = (token_types < 3)  # [B, L]

        neg_inf = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((L, L), neg_inf, device=device, dtype=dtype), diagonal=1
        )
        mask = causal.unsqueeze(0).expand(B, -1, -1).clone()  # [B, L, L]

        # Both positions are prompt → bidirectional (zero out the mask)
        prompt_i = is_prompt.unsqueeze(2).expand(-1, -1, L)
        prompt_j = is_prompt.unsqueeze(1).expand(-1, L, -1)
        mask[prompt_i & prompt_j] = 0.0

        # I-token attending to any prompt token → always allowed
        item_i = (~is_prompt).unsqueeze(2).expand(-1, -1, L)
        mask[item_i & prompt_j] = 0.0

        return mask


class TokenAwareLayerNorm(nn.Module):
    """Separate LayerNorm for each token type (U, O, E, I)."""

    def __init__(self, d_model: int, n_types: int = 4):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_types)
        ])

    def forward(self, x, token_types):
        out = torch.zeros_like(x)
        for t, norm in enumerate(self.norms):
            mask = (token_types == t)
            if mask.any():
                out[mask] = norm(x[mask])
        return out


class TokenAwareFFN(nn.Module):
    """Separate FFN for each token type (U, O, E, I)."""

    def __init__(self, d_model: int, d_ff: int, n_types: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_types)
        ])

    def forward(self, x, token_types):
        out = torch.zeros_like(x)
        for t, ffn in enumerate(self.ffns):
            mask = (token_types == t)
            if mask.any():
                out[mask] = ffn(x[mask])
        return out


# =========================================================================
# HSD Block  =  HSTU Attention  +  GPR enhancements
# =========================================================================

class HSDBlock(nn.Module):
    """
    Single HSD layer (Fig. 2b in GPR paper).

    Structure follows the paper diagram:
      1. HSTU Attention  (Q, K, V, U  +  hybrid mask)
      2. Token-Aware Norm & Add  (post-norm residual)
      3. Token-Aware FFN
      4. Token-Aware Norm & Add  (post-norm residual)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_types: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hstu_attn = HSTUAttention(d_model, n_heads, dropout)
        self.norm1 = TokenAwareLayerNorm(d_model, n_types)
        self.ffn = TokenAwareFFN(d_model, d_ff, n_types, dropout)
        self.norm2 = TokenAwareLayerNorm(d_model, n_types)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_types, seq_lens=None):
        # HSTU attention + residual + Token-Aware post-norm
        h = self.hstu_attn(x, token_types, seq_lens)
        x = self.norm1(x + self.dropout(h), token_types)

        # Token-Aware FFN + residual + Token-Aware post-norm
        h = self.ffn(x, token_types)
        x = self.norm2(x + self.dropout(h), token_types)

        return x


# =========================================================================
# MoR: Mixture-of-Recursions
# =========================================================================

class MixtureOfRecursions(nn.Module):
    """
    Mixture-of-Recursions (Bae et al. 2025, ref [3] in GPR paper).

    Each token is assigned a recursion depth d_i ∈ {0, 1, ..., R-1}
    by a learned router.  All tokens share the SAME block parameters
    but tokens with higher depth pass through the block more times,
    increasing effective model depth without adding parameters.
    """

    def __init__(self, d_model: int, max_recursions: int = 2):
        super().__init__()
        self.max_recursions = max_recursions
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, max_recursions),
        )

    def forward(self, x, block, token_types, seq_lens=None):
        B, L, D = x.shape
        R = self.max_recursions

        depth_logits = self.router(x.detach())  # stop-grad input
        if self.training:
            depth_probs = F.gumbel_softmax(depth_logits, tau=1.0, hard=True, dim=-1)
        else:
            depth_probs = F.one_hot(
                depth_logits.argmax(dim=-1), num_classes=R
            ).to(x.dtype)

        outputs = [x]
        current = x
        for r in range(R):
            current = block(current, token_types, seq_lens)
            outputs.append(current)

        stacked = torch.stack(outputs[1:], dim=2)  # [B, L, R, D]
        result = (stacked * depth_probs.unsqueeze(-1)).sum(dim=2)

        return result


# =========================================================================
# LLM Knowledge Module (Sec. 2.2 — External Knowledge)
# =========================================================================

class LLMKnowledgeModule(nn.Module):
    """
    External knowledge injection from a fine-tuned LLM (Sec. 2.2).

    In production GPR a real LLM generates a textual "thought process"
    about each user's potential interests, which is tokenized and
    integrated into the intent embeddings.  Here the module is an
    architectural placeholder that learns to generate thought tokens
    from the pooled intent.  Swapping in a frozen LLM encoder is a
    drop-in replacement of ``thought_generator``.
    """

    def __init__(self, d_model: int, n_thought_tokens: int = 4):
        super().__init__()
        self.n_thought_tokens = n_thought_tokens
        self.thought_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * n_thought_tokens),
        )
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, intent_summary):
        """
        intent_summary: [B, D] — mean-pooled HSD output
        Returns: thought_tokens [B, n_thought_tokens, D]
        """
        B = intent_summary.shape[0]
        raw = self.thought_generator(intent_summary)
        raw = raw.view(B, self.n_thought_tokens, -1)
        return self.projection(raw)


# =========================================================================
# HSD: Heterogeneous Sequence-wise Decoder
# =========================================================================

class HSD(nn.Module):
    """
    Heterogeneous Sequence-wise Decoder.

    Architecture:
      Input embedding (semantic IDs + user/env features + position + type)
        → N × HSDBlock
        → Mixture-of-Recursions
        → LLM Knowledge injection (concatenated thought tokens)
        → Final LayerNorm
        → intent embeddings  [B, L + K_thought, D]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers_hsd
        self.n_levels = cfg.n_semantic_levels

        # --- Input embeddings ---
        self.semantic_embed = nn.Embedding(
            cfg.codebook_size + 1, cfg.d_model, padding_idx=0
        )
        self.level_embed = nn.Embedding(cfg.n_semantic_levels, cfg.d_model)
        self.type_embed = nn.Embedding(cfg.n_token_types, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.user_proj = nn.Linear(cfg.d_user, cfg.d_model)
        self.env_proj = nn.Linear(cfg.d_env, cfg.d_model)

        self.level_agg = nn.Linear(cfg.d_model * cfg.n_semantic_levels, cfg.d_model)

        # --- Stack of HSD blocks ---
        self.blocks = nn.ModuleList([
            HSDBlock(cfg.d_model, cfg.n_heads, cfg.d_ff,
                     cfg.n_token_types, cfg.dropout)
            for _ in range(cfg.n_layers_hsd)
        ])

        # --- Mixture-of-Recursions ---
        self.mor = MixtureOfRecursions(cfg.d_model, cfg.n_mor_recursions)

        # --- External LLM knowledge ---
        self.llm_knowledge = LLMKnowledgeModule(
            cfg.d_model, cfg.n_llm_thought_tokens
        )

        self.final_norm = nn.LayerNorm(cfg.d_model)

    def _embed_input(self, semantic_ids, token_types, user_features, env_features):
        B, L, n_levels = semantic_ids.shape

        level_embeds = []
        for lvl in range(n_levels):
            sem = self.semantic_embed(semantic_ids[:, :, lvl])
            le = self.level_embed(
                torch.full((B, L), lvl, device=semantic_ids.device, dtype=torch.long)
            )
            level_embeds.append(sem + le)
        x = self.level_agg(torch.cat(level_embeds, dim=-1))

        u_mask = (token_types == 0)
        if u_mask.any():
            user_emb = self.user_proj(user_features)
            x = x + u_mask.unsqueeze(-1).to(x.dtype) * user_emb.unsqueeze(1)

        e_mask = (token_types == 2)
        if e_mask.any():
            env_emb = self.env_proj(env_features)
            x = x + e_mask.unsqueeze(-1).to(x.dtype) * env_emb.unsqueeze(1)

        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions) + self.type_embed(token_types)

        return x

    def forward(self, semantic_ids, token_types, user_features, env_features,
                seq_lens=None):
        x = self._embed_input(semantic_ids, token_types, user_features, env_features)

        for block in self.blocks:
            x = block(x, token_types, seq_lens)

        x = self.mor(x, self.blocks[-1], token_types, seq_lens)

        # Inject external LLM knowledge as additional thought tokens
        # that downstream PTD cross-attention can attend to (Sec. 2.2).
        intent_pool = x.mean(dim=1)                        # [B, D]
        thought_tokens = self.llm_knowledge(intent_pool)   # [B, K, D]
        x = torch.cat([x, thought_tokens], dim=1)         # [B, L+K, D]

        return self.final_norm(x)


# =========================================================================
# Refining Module — DDPM-style with cosine schedule (Fig. 2c)
# =========================================================================

class RefiningModule(nn.Module):
    """
    Diffusion-based refining module (Fig. 2c, ref [26]).

    Uses a cosine noise schedule (Improved DDPM) and ε-prediction:
      - Training: sample random t, noise target, predict noise
      - Inference: iterative DDPM posterior sampling from pure noise
    """

    def __init__(self, d_model: int, n_steps: int = 5, n_heads: int = 4):
        super().__init__()
        self.n_steps = n_steps
        self.d_model = d_model
        self.step_embed = nn.Embedding(n_steps, d_model)
        self.condition_proj = nn.Linear(d_model, d_model)

        self.denoiser = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=0.1,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, d_model)

        # Cosine schedule (Nichol & Dhariwal 2021)
        betas = self._cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _cosine_beta_schedule(n_steps, s=0.008):
        steps = torch.linspace(0, n_steps, n_steps + 1)
        ac = torch.cos(((steps / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _predict_noise(self, x_t, cond, t):
        step_emb = self.step_embed(t)
        inp = (x_t + cond + step_emb).unsqueeze(1)
        out = self.denoiser(inp).squeeze(1)
        return self.out_proj(out)

    def forward(self, condition, target=None):
        """
        condition: [B, D] — sum-pooled thinking tokens
        target:    [B, D] — ground-truth representation (training only)
        Returns:   (refined [B, D], aux_loss scalar)
        """
        B, D = condition.shape
        device = condition.device
        cond = self.condition_proj(condition)

        if self.training and target is not None:
            t = torch.randint(0, self.n_steps, (B,), device=device)
            noise = torch.randn_like(target)
            sqrt_a = self.sqrt_alphas_cumprod[t].unsqueeze(1)
            sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
            x_t = sqrt_a * target + sqrt_1ma * noise

            noise_pred = self._predict_noise(x_t, cond, t)
            aux_loss = F.mse_loss(noise_pred, noise)

            x_0_pred = (x_t - sqrt_1ma * noise_pred) / (sqrt_a + 1e-8)
            return x_0_pred, aux_loss
        else:
            x = torch.randn(B, D, device=device, dtype=cond.dtype)
            for step in reversed(range(self.n_steps)):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                noise_pred = self._predict_noise(x, cond, t)

                alpha_t = self.alphas_cumprod[step]
                x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / (
                    torch.sqrt(alpha_t) + 1e-8
                )
                if step > 0:
                    mean = (
                        self.posterior_mean_coef1[step] * x_0_pred
                        + self.posterior_mean_coef2[step] * x
                    )
                    var = self.posterior_var[step]
                    x = mean + torch.sqrt(var) * torch.randn_like(x)
                else:
                    x = x_0_pred
            return x, torch.tensor(0.0, device=device, dtype=cond.dtype)


# =========================================================================
# PTD: Progressive Token-wise Decoder
# =========================================================================

class PTD(nn.Module):
    """
    Progressive Token-wise Decoder.

    "Thinking → Refining → Generation" paradigm (Sec. 2.2).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_levels = cfg.n_semantic_levels
        self.codebook_size = cfg.codebook_size
        self.n_thinking = cfg.n_thinking_tokens

        self.thinking_queries = nn.Parameter(
            torch.randn(cfg.n_thinking_tokens, cfg.d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(cfg.d_model)

        self.refining = RefiningModule(
            cfg.d_model, cfg.n_refining_steps, cfg.n_heads
        )

        self.code_embed = nn.Embedding(cfg.codebook_size + 1, cfg.d_model)
        self.code_pos = nn.Embedding(cfg.n_semantic_levels, cfg.d_model)

        self.code_decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                batch_first=True,
            )
            for _ in range(cfg.n_layers_ptd)
        ])

        self.code_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.codebook_size)
            for _ in range(cfg.n_semantic_levels)
        ])

    def forward(self, intent_emb, target_codes=None):
        B = intent_emb.shape[0]
        device = intent_emb.device
        _dtype = intent_emb.dtype

        queries = self.thinking_queries.unsqueeze(0).expand(B, -1, -1)
        thinking, _ = self.cross_attn(queries, intent_emb, intent_emb)
        thinking = self.cross_norm(thinking + queries)

        condition = thinking.sum(dim=1)
        target_repr = self._codes_to_repr(target_codes) if target_codes is not None else None
        refined, refine_loss = self.refining(condition, target_repr)

        context = torch.cat([thinking, refined.unsqueeze(1)], dim=1)

        all_logits = []
        prev_code_emb = torch.zeros(B, 1, self.d_model, device=device, dtype=_dtype)

        for lvl in range(self.n_levels):
            pos_emb = self.code_pos(
                torch.full((B, 1), lvl, device=device, dtype=torch.long)
            )
            query = prev_code_emb + pos_emb

            for layer in self.code_decoder_layers:
                query = layer(query, context)

            logits = self.code_heads[lvl](query.squeeze(1))
            all_logits.append(logits)

            if target_codes is not None:
                next_code = target_codes[:, lvl]
            else:
                next_code = logits.argmax(dim=-1)
            prev_code_emb = self.code_embed(next_code).unsqueeze(1)

        return torch.stack(all_logits, dim=1), refine_loss

    def _codes_to_repr(self, codes):
        embs = [self.code_embed(codes[:, lvl]) for lvl in range(self.n_levels)]
        return torch.stack(embs, dim=1).mean(dim=1)

    @torch.no_grad()
    def generate(self, intent_emb, beam_width=1):
        B = intent_emb.shape[0]
        device = intent_emb.device
        _dtype = intent_emb.dtype

        queries = self.thinking_queries.unsqueeze(0).expand(B, -1, -1)
        thinking, _ = self.cross_attn(queries, intent_emb, intent_emb)
        thinking = self.cross_norm(thinking + queries)

        condition = thinking.sum(dim=1)
        refined, _ = self.refining(condition)
        context = torch.cat([thinking, refined.unsqueeze(1)], dim=1)

        codes = []
        prev_code_emb = torch.zeros(B, 1, self.d_model, device=device, dtype=_dtype)
        for lvl in range(self.n_levels):
            pos_emb = self.code_pos(
                torch.full((B, 1), lvl, device=device, dtype=torch.long)
            )
            query = prev_code_emb + pos_emb
            for layer in self.code_decoder_layers:
                query = layer(query, context)

            logits = self.code_heads[lvl](query.squeeze(1))
            code = logits.argmax(dim=-1)
            codes.append(code)
            prev_code_emb = self.code_embed(code).unsqueeze(1)

        return torch.stack(codes, dim=1)


# =========================================================================
# HTE: Hierarchical Token-wise Evaluator
# =========================================================================

class HTE(nn.Module):
    """
    Hierarchical Token-wise Evaluator.

    Predicts value at each semantic code level and a final aggregated
    value for auction and RL.  During RL post-training HTE serves as
    the critic (value function).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_levels = cfg.n_semantic_levels
        self.d_model = cfg.d_model

        self.code_embed = nn.Embedding(cfg.codebook_size + 1, cfg.d_model)

        self.level_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 1),
            )
            for _ in range(cfg.n_semantic_levels)
        ])

        self.final_value_head = nn.Sequential(
            nn.Linear(cfg.d_model * (cfg.n_semantic_levels + 1), cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, intent_summary, codes):
        """
        intent_summary: [B, D] — pooled intent from HSD
        codes:          [B, n_levels] — semantic codes
        Returns: (level_values [B, n_levels], final_value [B, 1])
        """
        level_values = []
        code_reprs = []

        for lvl in range(self.n_levels):
            code_emb = self.code_embed(codes[:, lvl])
            code_reprs.append(code_emb)
            combined = torch.cat([intent_summary, code_emb], dim=-1)
            level_values.append(self.level_predictors[lvl](combined))

        level_values = torch.cat(level_values, dim=-1)  # [B, n_levels]

        all_repr = torch.cat([intent_summary] + code_reprs, dim=-1)
        final_value = self.final_value_head(all_repr)    # [B, 1]

        return level_values, final_value


# =========================================================================
# SemanticTrie — for Value-Guided Trie-Based Beam Search (Sec. 2.3)
# =========================================================================

class TrieNode:
    __slots__ = ["children", "is_terminal", "item_ids"]

    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.is_terminal = False
        self.item_ids: list = []


class SemanticTrie:
    """
    Trie over semantic ID paths.

    Used during Value-Guided Trie-Based Beam Search (Sec. 2.3) to
    constrain generation to valid item code paths and enable early
    pruning based on user targeting.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, code_path, item_id=None):
        node = self.root
        for code in code_path:
            c = int(code)
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_terminal = True
        if item_id is not None:
            node.item_ids.append(item_id)

    def get_valid_children(self, prefix):
        """Return set of valid next codes given a prefix."""
        node = self.root
        for code in prefix:
            c = int(code)
            if c not in node.children:
                return set()
            node = node.children[c]
        return set(node.children.keys())

    @classmethod
    def build_from_items(cls, item2sid: dict) -> "SemanticTrie":
        trie = cls()
        for item_id, sid in item2sid.items():
            trie.insert(sid, item_id)
        return trie


# =========================================================================
# GPR: Full Model
# =========================================================================

class GPR(nn.Module):
    """
    GPR: Generative Pre-trained Recommender.

    End-to-end pipeline:
      HSD (user understanding)  →  PTD (item generation)  →  HTE (value estimation)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hsd = HSD(cfg)
        self.ptd = PTD(cfg)
        self.hte = HTE(cfg)

        self.n_mtp_heads = cfg.n_mtp_heads
        self.mtp_projections = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.d_model)
            for _ in range(cfg.n_mtp_heads)
        ])

    def forward(self, batch, mode="mtp", **kwargs):
        # Modes that handle their own HSD call (return early)
        if mode == "hepo_candidates":
            return self._forward_hepo_candidates(batch, **kwargs)

        semantic_ids = batch["semantic_ids"]
        token_types = batch["token_types"]
        user_features = batch["user_features"]
        env_features = batch["env_features"]
        seq_lens = batch["seq_len"]
        target_ids = batch["target_ids"]

        intent = self.hsd(
            semantic_ids, token_types, user_features, env_features, seq_lens
        )

        intent_summary = intent.mean(dim=1)

        if mode == "mtp":
            return self._forward_mtp(intent, intent_summary, target_ids)
        elif mode == "vaft":
            return self._forward_vaft(intent, intent_summary, target_ids, batch)
        elif mode == "hepo_generate":
            return self._forward_generate(intent, intent_summary)
        elif mode == "hepo_train":
            return self._forward_hepo_train(batch, intent, intent_summary)
        else:
            return self._forward_mtp(intent, intent_summary, target_ids)

    def _forward_mtp(self, intent, intent_summary, target_ids):
        all_logits = []
        total_refine_loss = 0.0

        for head_idx in range(self.n_mtp_heads):
            projected = self.mtp_projections[head_idx](intent)
            logits, refine_loss = self.ptd(projected, target_ids)
            all_logits.append(logits)
            total_refine_loss += refine_loss

        with torch.no_grad():
            pred_codes = all_logits[0].argmax(dim=-1)

        # stopgrad(h) for value function (Sec. 3.3)
        level_values, final_value = self.hte(intent_summary.detach(), target_ids)

        return {
            "all_logits": all_logits,
            "refine_loss": total_refine_loss / self.n_mtp_heads,
            "level_values": level_values,
            "final_value": final_value,
            "pred_codes": pred_codes,
        }

    def _forward_vaft(self, intent, intent_summary, target_ids, batch):
        result = self._forward_mtp(intent, intent_summary, target_ids)
        result["action_types"] = batch["action_types"]
        result["values"] = batch["values"]
        result["target_value"] = batch["target_value"]
        result["target_action"] = batch["target_action"]
        return result

    def _forward_generate(self, intent, intent_summary):
        projected = self.mtp_projections[0](intent)
        codes = self.ptd.generate(projected)
        # stopgrad(h) for value function (Sec. 3.3)
        level_values, final_value = self.hte(intent_summary.detach(), codes)
        return {"codes": codes, "level_values": level_values, "final_value": final_value}

    def _forward_hepo_candidates(self, batch, n_candidates=20):
        """Generate K candidates via forward() so FSDP gathers params."""
        with torch.no_grad():
            return self.generate_candidates(batch, n_candidates=n_candidates)

    def _forward_hepo_train(self, batch, intent, intent_summary):
        """Compute new log-probs and HTE values for HEPO loss.

        Expects batch to contain 'cand_codes' [B, K, n_levels] from
        a prior hepo_candidates call.  Returns new_logprobs (with grad
        through HSD/PTD) and value_preds (with grad through HTE).
        """
        cand_codes = batch["cand_codes"]
        B, K, n_levels = cand_codes.shape
        device = cand_codes.device

        intent_summary_det = intent_summary.detach()

        all_new_logprobs = []
        for k_idx in range(K):
            codes_k = cand_codes[:, k_idx, :]
            projected = self.mtp_projections[k_idx % self.n_mtp_heads](intent)
            logits, _ = self.ptd(projected, codes_k)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(2, codes_k.unsqueeze(-1)).squeeze(-1)
            all_new_logprobs.append(selected)
        new_logprobs = torch.stack(all_new_logprobs, dim=1)

        value_preds = torch.zeros(
            B, K, n_levels, device=device, dtype=intent.dtype,
        )
        for k_idx in range(K):
            codes_k = cand_codes[:, k_idx, :]
            for lvl in range(n_levels):
                partial = codes_k.clone()
                partial[:, lvl:] = 0
                _, fv = self.hte(intent_summary_det, partial)
                value_preds[:, k_idx, lvl] = fv.squeeze(-1)

        return {
            "new_logprobs": new_logprobs,
            "value_preds": value_preds,
        }

    # -----------------------------------------------------------------
    # Candidate generation (sampling-based, used when no trie)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate_candidates(self, batch, n_candidates=20):
        """Generate K candidate items for HEPO simulation."""
        semantic_ids = batch["semantic_ids"]
        token_types = batch["token_types"]
        user_features = batch["user_features"]
        env_features = batch["env_features"]
        seq_lens = batch["seq_len"]

        intent = self.hsd(
            semantic_ids, token_types, user_features, env_features, seq_lens
        )
        intent_summary = intent.mean(dim=1)

        B = intent.shape[0]
        all_codes, all_values, all_logprobs = [], [], []

        for k in range(n_candidates):
            head_idx = k % self.n_mtp_heads
            projected = self.mtp_projections[head_idx](intent)

            queries = self.ptd.thinking_queries.unsqueeze(0).expand(B, -1, -1)
            thinking, _ = self.ptd.cross_attn(queries, projected, projected)
            thinking = self.ptd.cross_norm(thinking + queries)

            condition = thinking.sum(dim=1)
            noise = torch.randn_like(condition) * (0.1 + 0.05 * k)
            refined, _ = self.ptd.refining(condition + noise)

            context = torch.cat([thinking, refined.unsqueeze(1)], dim=1)

            codes_k, logprobs_k = [], []
            prev = torch.zeros(B, 1, self.ptd.d_model, device=intent.device, dtype=intent.dtype)

            for lvl in range(self.ptd.n_levels):
                pos_emb = self.ptd.code_pos(
                    torch.full((B, 1), lvl, device=intent.device, dtype=torch.long)
                )
                query = prev + pos_emb
                for layer in self.ptd.code_decoder_layers:
                    query = layer(query, context)

                logits = self.ptd.code_heads[lvl](query.squeeze(1))

                temperature = 0.8 + 0.1 * k
                probs = F.softmax(logits / temperature, dim=-1)
                code = torch.multinomial(probs, 1).squeeze(-1)
                lp = torch.log(probs.gather(1, code.unsqueeze(1)) + 1e-10).squeeze(-1)

                codes_k.append(code)
                logprobs_k.append(lp)
                prev = self.ptd.code_embed(code).unsqueeze(1)

            codes_k = torch.stack(codes_k, dim=1)
            logprobs_k = torch.stack(logprobs_k, dim=1)

            _, fv = self.hte(intent_summary, codes_k)
            all_codes.append(codes_k)
            all_values.append(fv.squeeze(-1))
            all_logprobs.append(logprobs_k)

        return {
            "codes": torch.stack(all_codes, dim=1),
            "values": torch.stack(all_values, dim=1),
            "logprobs": torch.stack(all_logprobs, dim=1),
        }

    # -----------------------------------------------------------------
    # Trie-constrained value-guided beam search (Sec. 2.3)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def trie_beam_search(self, batch, trie: SemanticTrie,
                         beam_width: int = 10, n_results: int = 20):
        """
        Value-Guided Trie-Based Beam Search (Sec. 2.3).

        At each semantic level the search is constrained to codes that
        exist in *trie*.  HTE value estimates dynamically expand the
        beam for high-value prefixes and prune low-value ones.

        Operates per-sample (loop over B) because each sample may have
        a different set of valid trie paths.
        """
        semantic_ids = batch["semantic_ids"]
        token_types = batch["token_types"]
        user_features = batch["user_features"]
        env_features = batch["env_features"]
        seq_lens = batch["seq_len"]

        intent = self.hsd(
            semantic_ids, token_types, user_features, env_features, seq_lens
        )
        intent_summary = intent.mean(dim=1)
        B = intent.shape[0]
        n_levels = self.ptd.n_levels
        device = intent.device

        all_result_codes = []
        all_result_values = []

        for b in range(B):
            intent_b = intent[b:b+1]                    # [1, L, D]
            isumm_b = intent_summary[b:b+1]             # [1, D]

            projected = self.mtp_projections[0](intent_b)
            queries = self.ptd.thinking_queries.unsqueeze(0)
            thinking, _ = self.ptd.cross_attn(queries, projected, projected)
            thinking = self.ptd.cross_norm(thinking + queries)
            condition = thinking.sum(dim=1)
            refined, _ = self.ptd.refining(condition)
            context = torch.cat([thinking, refined.unsqueeze(1)], dim=1)  # [1, K+1, D]

            # Each beam entry: (prefix_codes: list[int], cumulative_log_prob: float,
            #                    prev_emb: Tensor [1,1,D])
            beams = [([], 0.0, torch.zeros(1, 1, self.ptd.d_model, device=device, dtype=intent.dtype))]

            for lvl in range(n_levels):
                new_beams = []
                for prefix, cum_lp, prev_emb in beams:
                    valid_codes = trie.get_valid_children(prefix)
                    if not valid_codes:
                        continue

                    pos_emb = self.ptd.code_pos(
                        torch.full((1, 1), lvl, device=device, dtype=torch.long)
                    )
                    query = prev_emb + pos_emb
                    for layer in self.ptd.code_decoder_layers:
                        query = layer(query, context)

                    logits = self.ptd.code_heads[lvl](query.squeeze(1))   # [1, C]
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [C]

                    # Get HTE level value for dynamic beam width
                    for c in valid_codes:
                        c_tensor = torch.tensor([[c]], device=device, dtype=torch.long)
                        partial = torch.zeros(1, n_levels, device=device, dtype=torch.long)
                        for i, pc in enumerate(prefix):
                            partial[0, i] = pc
                        partial[0, lvl] = c
                        lv, _ = self.hte(isumm_b, partial)
                        value_bonus = lv[0, lvl].item() * 0.1
                        new_beams.append((
                            prefix + [c],
                            cum_lp + log_probs[c].item() + value_bonus,
                            self.ptd.code_embed(c_tensor).unsqueeze(0).view(1, 1, -1),
                        ))

                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

            # Collect final results with values
            result_codes_b = []
            result_values_b = []
            for prefix, _, _ in beams[:n_results]:
                code_t = torch.tensor([prefix], device=device, dtype=torch.long)
                _, fv = self.hte(isumm_b, code_t)
                result_codes_b.append(code_t.squeeze(0))
                result_values_b.append(fv.squeeze())

            while len(result_codes_b) < n_results:
                result_codes_b.append(torch.zeros(n_levels, device=device, dtype=torch.long))
                result_values_b.append(torch.tensor(0.0, device=device, dtype=intent.dtype))

            all_result_codes.append(torch.stack(result_codes_b[:n_results]))
            all_result_values.append(torch.stack(result_values_b[:n_results]))

        return {
            "codes": torch.stack(all_result_codes),    # [B, n_results, n_levels]
            "values": torch.stack(all_result_values),  # [B, n_results]
        }


# =========================================================================
# Loss functions
# =========================================================================

def mtp_loss(result, target_ids, n_heads=4):
    """
    Multi-Token Prediction loss (Eq. 3 in GPR paper).
    Aggregates per-head, per-level CE with uniform head weights.
    """
    all_logits = result["all_logits"]
    n_levels = target_ids.shape[1]

    total_ce = 0.0
    w_h = 1.0 / n_heads

    for logits in all_logits:
        for lvl in range(n_levels):
            total_ce += w_h * F.cross_entropy(logits[:, lvl, :], target_ids[:, lvl])

    total = total_ce + 0.1 * result["refine_loss"]
    return total, {
        "ce_loss": total_ce.item(),
        "refine_loss": (
            result["refine_loss"].item()
            if isinstance(result["refine_loss"], torch.Tensor)
            else result["refine_loss"]
        ),
    }


def vaft_loss(result, target_ids, target_values, target_actions):
    """
    Value-Aware Fine-Tuning loss (Eq. 4 in GPR paper).
    Reweights per-position MTP loss by action type and eCPM:
      impression (w=1), click (w=eCPM/pCTR≈2), conversion (w=eCPM/(pCTR×pCVR)≈4)
    """
    all_logits = result["all_logits"]
    n_heads = len(all_logits)
    n_levels = target_ids.shape[1]
    device = target_ids.device

    action_weights = torch.tensor([1.0, 2.0, 4.0], device=device, dtype=target_values.dtype)
    w_action = action_weights[target_actions]

    v_norm = target_values / (target_values.mean() + 1e-8)
    w_value = w_action * v_norm.clamp(0.1, 10.0)

    total_ce = 0.0
    w_h = 1.0 / n_heads

    for logits in all_logits:
        for lvl in range(n_levels):
            ce = F.cross_entropy(logits[:, lvl, :], target_ids[:, lvl], reduction="none")
            total_ce += w_h * (ce * w_value).mean()

    value_loss = F.mse_loss(result["final_value"].squeeze(-1), target_values)

    total = total_ce + 0.1 * result["refine_loss"] + 0.1 * value_loss
    return total, {"ce_loss": total_ce.item(), "value_loss": value_loss.item()}
