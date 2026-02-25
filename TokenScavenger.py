# Bismillahi Rahmani Rahim

# Author Maulidi Barasa
# ==============================================================================
# DeepMime Trainer — LAA-GIGPO Integration
# ==============================================================================
"""
DEEP-MIME: Deep Exploration and Exploitation Policy with Memory-Integrated
Model Emulation.

This module implements a reinforcement learning trainer that combines:
  - LAA-GiGPO: Two-level credit assignment with latent group advantages
    (micro-advantage within groups, macro-advantage across groups)
  - Carousel Memory Alignment: MSE loss between current and historical 
    log-probabilities on replay sequences
  - Adaptive entropy control to prevent collapse
  - Shaped rewards with whitespace penalty, diversity bonus, and length bonus
  - Differentiable memory with MoE routing

The trainer supports weight-encoded teacher-student knowledge distillation
with hybrid sparse/PCA weight encoders and global weight statistics.
"""

import gc
import math
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    """Central, read-only configuration namespace for DeepMime.

    All hyper-parameters are class-level attributes so they can be read
    without instantiation.  Dynamic teacher-architecture selection is
    handled via class-methods.
    """

    # ── Teacher architecture selection ──────────────────────────────────────
    TEACHER_ARCH: str = "1.3b"

    _TEACHER_MODEL_ID_1_3B: str   = "deepseek-ai/deepseek-coder-1.3b-instruct"
    _TEACHER_HIDDEN_DIM_1_3B: int = 2048
    _TEACHER_NUM_LAYERS_1_3B: int = 24
    _TEACHER_VOCAB_SIZE_1_3B: int = 32256

    _TEACHER_MODEL_ID_33B: str   = "deepseek-ai/deepseek-coder-33b-instruct"
    _TEACHER_HIDDEN_DIM_33B: int = 7168
    _TEACHER_NUM_LAYERS_33B: int = 62
    _TEACHER_VOCAB_SIZE_33B: int = 32256

    @classmethod
    def get_teacher_model_id(cls) -> str:
        """Return the HuggingFace model identifier for the selected teacher."""
        return (
            cls._TEACHER_MODEL_ID_1_3B
            if cls.TEACHER_ARCH == "1.3b"
            else cls._TEACHER_MODEL_ID_33B
        )

    @classmethod
    def get_teacher_hidden_dim(cls) -> int:
        """Return the hidden dimension of the selected teacher architecture."""
        return (
            cls._TEACHER_HIDDEN_DIM_1_3B
            if cls.TEACHER_ARCH == "1.3b"
            else cls._TEACHER_HIDDEN_DIM_33B
        )

    @classmethod
    def get_teacher_num_layers(cls) -> int:
        """Return the number of transformer layers in the selected teacher."""
        return (
            cls._TEACHER_NUM_LAYERS_1_3B
            if cls.TEACHER_ARCH == "1.3b"
            else cls._TEACHER_NUM_LAYERS_33B
        )

    @classmethod
    def get_vocab_size(cls) -> int:
        """Return the shared vocabulary size (identical across both teachers)."""
        return cls._TEACHER_VOCAB_SIZE_1_3B

    # ── Weight encoder ───────────────────────────────────────────────────────
    WEIGHT_MATRICES_TO_ENCODE: List[str] = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "mlp.gate_proj.weight",
    ]
    WEIGHT_ENCODER_STRATEGY: str      = "hybrid"
    TOP_K_SPARSE_WEIGHTS: int         = 64
    WEIGHT_ENCODER_SPARSE_HIDDEN: int = 64
    PCA_RANK: int                     = 16
    PCA_BLOCK_SIZE: int               = 512
    WEIGHT_ENCODER_PCA_HIDDEN: int    = 64
    WEIGHT_EMB_DIM: int               = 64
    GLOBAL_STATS_DIM: int             = 5
    WEIGHT_ENCODER_HIDDEN: int        = 512
    WEIGHT_ENCODER_MAX_FLAT: int      = 1024

    # ── ATLAS-GRPO / LUMOS completion predicates ─────────────────────────────
    _MINIMUM_CONTENT_CHARS: int = 25
    _MINIMUM_UNIQUE_CHARS: int  = 50
    _MINIMUM_GEN_TOKENS: int    = 75

    _SENTENCE_ENDINGS: str      = ".!?;}"

    # ── Student architecture ──────────────────────────────────────────────────
    HIDDEN_DIM: int       = 1024
    INTERMEDIATE_DIM: int = 512
    NUM_LAYERS: int       = 6
    NUM_HEADS: int        = 4
    LATENT_DIM: int       = 128

    # Rotary position embedding dimension
    ROPE_DIM: int         = 128

    NUM_EXPERTS: int               = 4
    ACTIVE_EXPERTS: int            = 2
    LOAD_BALANCE_LOSS_WEIGHT: float = 0.01
    MEMORY_SLOTS: int              = 64
    KEY_DIM: int                   = 64
    VALUE_DIM: int                 = 64
    MEMORY_TOP_K: int              = 2

    # ── Regularisation & training ─────────────────────────────────────────────
    ORTHOGONALITY_LAMBDA: float   = 0.01
    CONTRASTIVE_TEMPERATURE: float = 0.125
    LEARNING_RATE: float          = 5.0e-4
    DISCRIMINATOR_HIDDEN_DIM: int = 256
    GAMMA_GAIL: float             = 0.1
    DISCOUNT_GAMMA: float         = 0.99
    GAE_LAMBDA: float             = 0.95
    PPO_CLIP_EPSILON: float       = 0.2
    VALUE_MLP_DIM: int            = 256
    ALPHA_SUPERVISED: float       = 0.5
    BETA_HIDDEN: float            = 0.5
    POLICY_GRADIENT_WEIGHT: float = 1.0
    DAGGER_BETA_START: float      = 1.0
    DAGGER_BETA_DECAY: float      = 0.995
    DAGGER_BETA_MIN: float        = 0.1

    SEQ_LEN: int    = 512
    BATCH_SIZE: int = 1
    DEVICE: str     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Trainer-level constants ─────────────────────────────────────────────────
EMA_DECAY: float               = 0.995
REPLAY_BUFFER_CAPACITY: int    = 500
LAGRANGIAN_LR: float           = 0.01
EMPTY_TEXT_PENALTY: float      = 5.0
REPETITION_PENALTY: float      = 5.0
HIDDEN_LOSS_SCALE: float       = 0.01
ILP_COVERAGE_THRESHOLD: float  = 0.5
ADVANTAGE_CLAMP_MAX: float     = 5.0
ROLLOUT_GROUP_SIZE: int        = 4     # G — total rollouts per prompt step

# ── LAA-GiGPO policy constants ───────────────────────────────────────────────
# K: number of latent groups.  The online_group_size rollouts are partitioned
# into K groups of M = online_group_size // K sequences each.  Micro-advantage
# normalises within each group; macro-advantage normalises across groups.
GIGPO_LATENT_GROUP_SIZE: int       = 4
# Weight of the Carousel Memory Alignment loss: α_align * MSE(π_θ_lp, π_old_lp)
# on the replay portion of the augmented batch.  Prevents policy from drifting
# away from historically successful trajectories stored in the experience buffer.
GIGPO_CAROUSEL_ALIGN_WEIGHT: float = 0.1

# Generation temperature for exploration
ONLINE_GENERATION_TEMPERATURE: float = 0.875

# KL penalty coefficient (0.0 matches DAPO / Dr.GRPO recommendation)
BETA_GRPO: float = 0.0

# Loss weights — CE anchors distribution, GRPO provides gentle improvement
CE_LOSS_WEIGHT: float     = 1.0
GRPO_LOSS_WEIGHT: float   = 0.25
HIDDEN_LOSS_WEIGHT: float = 0.275

# RL warmup — pure CE for first N steps, then ramp to full GRPO weight
RL_WARMUP_STEPS: int       = 100
RL_MAX_GRPO_WEIGHT: float  = 0.2  # ceiling weight after full ramp

# Adaptive entropy controller target — prevents entropy collapse
ENTROPY_TARGET_NATS: float = 1.5   # ≈ 15 % of log(32768) ≈ 10.4 nats
ENTROPY_ALPHA_INIT: float  = 0.1   # starting entropy-bonus coefficient
ENTROPY_ALPHA_DELTA: float = 0.05  # per-step increment when below target

# Shaped reward components — debias reward landscape
WS_PENALTY_COEFF: float      = 10.0  # penalty for whitespace-fraction > 50 %
DIVERSITY_BONUS_COEFF: float  = 0.5   # bonus for unique-token ratio
LENGTH_BONUS_MAX: float       = 0.5   # maximum length bonus (saturates at 20 tokens)

# MoE routing stabilisation — soft routing and gate jitter for expert stability
MOE_LOAD_BALANCE_COEFF: float = 0.05  # elevated load-balance for 2-expert fragility
MOE_GATE_NOISE_STD: float     = 0.1   # gate-logit noise std to prevent lock-in

# Memory and training stability parameters
MEMORY_OUTPUT_SCALE: float = 0.1    # scale memory delta to prevent residual domination
WEIGHT_DECAY: float        = 0.001  # lets RL signal update freely
GRAD_ACCUM_STEPS: int      = 4      # fresher, lower-variance gradients

# ==============================================================================
# 2. WEIGHT ENCODERS & CACHE
# ==============================================================================
class WeightEncoderStrategy(str, Enum):
    """Supported strategies for encoding a weight matrix into a fixed-dim vector."""

    SPARSE = "sparse"
    PCA    = "pca"
    HYBRID = "hybrid"


class SparseWeightEncoder(nn.Module):
    """Encode a weight matrix using its top-K magnitude elements.

    The encoder extracts normalised values, row positions, and column positions
    of the top-K absolute-value entries and projects them through a gated MLP
    to produce a fixed-dimensional embedding.

    Args:
        top_k:           Number of top-magnitude elements to retain.
        encoder_hidden:  Hidden width of the gated projection.
        emb_dim:         Output embedding dimension.
        num_layers:      Model depth used for depth-aware weight scaling.
    """

    def __init__(
        self,
        top_k: int,
        encoder_hidden: int,
        emb_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.top_k           = top_k
        self._feat_dim       = top_k * 3
        self._encoder_hidden = encoder_hidden
        self.gate_up = nn.Linear(self._feat_dim, encoder_hidden * 2, bias=True)
        self.down    = nn.Linear(encoder_hidden, emb_dim, bias=True)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "_feat_buf", torch.zeros(self._feat_dim), persistent=False
        )
        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int = 1) -> None:
        """Initialise linear layers with orthogonal init and depth scaling.

        Args:
            num_layers: Number of student layers for depth-aware output scaling.
        """
        depth_scale = 1.0 / math.sqrt(2.0 * max(num_layers, 1))
        nn.init.orthogonal_(self.gate_up.weight, gain=math.sqrt(2))
        nn.init.constant_(self.gate_up.bias, 0.0)
        nn.init.orthogonal_(self.down.weight, gain=1.0)
        nn.init.constant_(self.down.bias, 0.0)
        with torch.no_grad():
            self.down.weight.mul_(depth_scale)

    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Encode weight_matrix into a fixed-dim embedding.

        Args:
            weight_matrix: Arbitrary-shape weight tensor.

        Returns:
            Tensor of shape (emb_dim,).

        Time complexity:  O(N) for topk on N = numel(weight_matrix).
        Space complexity: O(top_k).
        """
        weight_f32   = weight_matrix.detach().float()
        num_rows     = weight_f32.shape[0]
        num_cols     = weight_f32.numel() // num_rows
        flat         = weight_f32.reshape(-1)
        num_elements = flat.numel()
        effective_k  = min(self.top_k, num_elements)

        _, topk_indices = torch.topk(flat.abs(), effective_k, largest=True, sorted=False)
        topk_values     = flat[topk_indices]

        row_norm      = topk_indices.float().div(num_cols).floor_().div_(max(num_rows - 1, 1))
        col_norm      = topk_indices.float().remainder(num_cols).div_(max(num_cols - 1, 1))
        value_mean    = topk_values.mean()
        value_std     = topk_values.std(correction=0).clamp_(min=1e-8)
        normed_values = (topk_values - value_mean).div_(value_std)

        buf = self._feat_buf
        buf.zero_()
        buf.narrow(0, 0,              effective_k).copy_(normed_values)
        buf.narrow(0, self.top_k,     effective_k).copy_(row_norm)
        buf.narrow(0, self.top_k * 2, effective_k).copy_(col_norm)

        h    = self.gate_up(buf)
        gate = h[: self._encoder_hidden]
        up   = h[self._encoder_hidden:]
        mid  = F.silu(gate, inplace=True) * up
        out  = self.down(mid)
        scale = self.log_scale.clamp(-4.0, 4.0).exp()
        return out * scale


class PCAWeightEncoder(nn.Module):
    """Encode a weight matrix via truncated SVD (PCA) of a random sub-block.

    Args:
        pca_rank:       Number of singular values / vectors to retain.
        pca_block_size: Maximum row/column size of the SVD sub-block.
        encoder_hidden: Hidden width of the projection MLP.
        emb_dim:        Output embedding dimension.
    """

    def __init__(
        self,
        pca_rank: int,
        pca_block_size: int,
        encoder_hidden: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.pca_rank       = pca_rank
        self.pca_block_size = pca_block_size
        feature_dim = 3 * pca_rank
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, encoder_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(encoder_hidden, emb_dim, bias=True),
        )

    def _block_sample(self, weight_f32: torch.Tensor) -> torch.Tensor:
        """Randomly sub-sample rows and columns to at most pca_block_size each.

        Args:
            weight_f32: 2-D float32 weight tensor.

        Returns:
            Sub-sampled 2-D tensor.
        """
        m, n      = weight_f32.shape
        row_limit = min(m, self.pca_block_size)
        col_limit = min(n, self.pca_block_size)
        row_perm  = torch.randperm(m, device=weight_f32.device)[:row_limit]
        col_perm  = torch.randperm(n, device=weight_f32.device)[:col_limit]
        return weight_f32[row_perm][:, col_perm]

    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Encode weight_matrix into a fixed-dim embedding via PCA features.

        Args:
            weight_matrix: Arbitrary-shape weight tensor.

        Returns:
            Tensor of shape (emb_dim,).  All-zeros on SVD failure.

        Time complexity:  O(min(m,n) * pca_block_size^2) for truncated SVD.
        Space complexity: O(pca_block_size^2).
        """
        weight_f32 = weight_matrix.detach().float()
        if weight_f32.dim() == 1:
            weight_f32 = weight_f32.unsqueeze(0)
        elif weight_f32.dim() > 2:
            weight_f32 = weight_f32.reshape(weight_f32.shape[0], -1)

        block          = self._block_sample(weight_f32)
        effective_rank = min(self.pca_rank, block.shape[0], block.shape[1])

        try:
            U, S, Vt = torch.linalg.svd(block, full_matrices=False)
        except torch.linalg.LinAlgError:
            logging.getLogger("RL-ILP-Agent").warning(
                "SVD failed on weight block; returning zero embedding."
            )
            return torch.zeros(self.mlp[-1].out_features, device=weight_matrix.device)

        def _pad_to_rank(t: torch.Tensor, target: int) -> torch.Tensor:
            if t.shape[0] >= target:
                return t[:target]
            pad_shape = (target - t.shape[0],) + t.shape[1:]
            return torch.cat([t, torch.zeros(pad_shape, device=t.device)], dim=0)

        sv       = _pad_to_rank(S, self.pca_rank)
        sv       = sv / sv[0].clamp(min=1e-8)
        u_norms  = _pad_to_rank(U.t(), self.pca_rank).norm(dim=1)
        vt_norms = _pad_to_rank(Vt, self.pca_rank).norm(dim=1)
        features = torch.cat([sv, u_norms, vt_norms], dim=0)
        return self.mlp(features)


def _compute_global_weight_stats(weight_2d: torch.Tensor) -> torch.Tensor:
    """Compute five global statistics for a 2-D weight matrix.

    Statistics: log1p Frobenius norm, mean, std, log1p max-abs, effective-rank ratio.

    Args:
        weight_2d: 2-D float weight tensor.

    Returns:
        Tensor of shape (5,).

    Time complexity:  O(N + S^2) where S = min(STAT_BLOCK, min(m, n)).
    Space complexity: O(S^2).
    """
    flat      = weight_2d.reshape(-1)
    frob_norm = torch.log1p(weight_2d.norm())
    elem_mean = flat.mean()
    elem_std  = flat.std().clamp(min=1e-8)
    max_abs   = torch.log1p(flat.abs().max())

    STAT_BLOCK: int = 256
    row_limit   = min(weight_2d.shape[0], STAT_BLOCK)
    col_limit   = min(weight_2d.shape[1], STAT_BLOCK)
    row_idx     = torch.randperm(weight_2d.shape[0], device=weight_2d.device)[:row_limit]
    col_idx     = torch.randperm(weight_2d.shape[1], device=weight_2d.device)[:col_limit]
    mini_block  = weight_2d[row_idx][:, col_idx]

    try:
        _, singular_vals, _ = torch.linalg.svd(mini_block, full_matrices=False)
        s_probs    = singular_vals / singular_vals.sum().clamp(min=1e-8)
        entropy    = -(s_probs * (s_probs + 1e-8).log()).sum()
        eff_rank_ratio = entropy.exp() / max(
            min(weight_2d.shape[0], weight_2d.shape[1]), 1
        )
    except torch.linalg.LinAlgError:
        eff_rank_ratio = torch.zeros(1, device=weight_2d.device).squeeze()

    return torch.stack([frob_norm, elem_mean, elem_std, max_abs, eff_rank_ratio])


class HybridWeightEncoder(nn.Module):
    """Combine sparse top-K and PCA embeddings with global statistics.

    The final embedding is the output of a fusion head that receives the
    concatenation of a sparse sub-embedding, a PCA sub-embedding, and a
    five-dimensional global statistics vector.

    Args:
        top_k:           Top-K element count for the sparse branch.
        pca_rank:        PCA rank for the SVD branch.
        pca_block_size:  Maximum sub-block size for the SVD branch.
        sparse_hidden:   Hidden width of the sparse MLP.
        pca_hidden:      Hidden width of the PCA MLP.
        emb_dim:         Output embedding dimension (must be even).
        global_stats_dim: Dimension of global statistics (default 5).
    """

    _GLOBAL_STATS_DIM: int = 5

    def __init__(
        self,
        top_k: int,
        pca_rank: int,
        pca_block_size: int,
        sparse_hidden: int,
        pca_hidden: int,
        emb_dim: int,
        global_stats_dim: int = 5,
    ) -> None:
        super().__init__()
        self.top_k           = top_k
        self.pca_rank        = pca_rank
        self.pca_block_size  = pca_block_size
        self.emb_dim         = emb_dim
        self.global_stats_dim = global_stats_dim
        half_emb = emb_dim // 2

        sparse_in = top_k * 3
        self.sparse_mlp = nn.Sequential(
            nn.Linear(sparse_in, sparse_hidden),
            nn.LayerNorm(sparse_hidden),
            nn.GELU(),
            nn.Linear(sparse_hidden, half_emb),
        )

        pca_in = pca_rank * 3
        self.pca_mlp = nn.Sequential(
            nn.Linear(pca_in, pca_hidden),
            nn.LayerNorm(pca_hidden),
            nn.GELU(),
            nn.Linear(pca_hidden, half_emb),
        )

        fusion_in = half_emb + half_emb + global_stats_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    @staticmethod
    def _to_2d(weight_tensor: torch.Tensor) -> torch.Tensor:
        """Reshape weight_tensor to 2-D (rows × columns).

        Args:
            weight_tensor: Tensor of any shape.

        Returns:
            2-D tensor.
        """
        if weight_tensor.dim() == 1:
            return weight_tensor.unsqueeze(0)
        if weight_tensor.dim() > 2:
            return weight_tensor.reshape(weight_tensor.shape[0], -1)
        return weight_tensor

    def _compute_sparse_features(
        self, weight_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Extract top-K magnitude features from a flat weight vector.

        Args:
            weight_flat: 1-D weight tensor.

        Returns:
            Tuple of (feature_vector of shape (top_k * 3,), validity_ratio).
        """
        numel    = weight_flat.numel()
        k        = min(self.top_k, numel)
        validity = k / self.top_k
        abs_w    = weight_flat.abs()
        vals, idx = torch.topk(abs_w, k)
        v_max      = vals[0].clamp(min=1e-8)
        norm_vals  = vals / v_max
        norm_idx   = idx.float() / max(numel - 1, 1)
        signs      = weight_flat[idx].sign()
        if k < self.top_k:
            pad_len   = self.top_k - k
            norm_vals = F.pad(norm_vals, (0, pad_len))
            norm_idx  = F.pad(norm_idx,  (0, pad_len))
            signs     = F.pad(signs,     (0, pad_len))
        feature_vec = torch.cat([norm_vals, norm_idx, signs], dim=0)
        return feature_vec, validity

    def _compute_pca_features(
        self, weight_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Extract PCA features from a 2-D weight matrix.

        Args:
            weight_2d: 2-D float tensor.

        Returns:
            Tuple of (feature_vector of shape (pca_rank * 3,), validity_ratio).
        """
        m, n   = weight_2d.shape
        device = weight_2d.device

        if m > self.pca_block_size:
            row_idx = torch.randperm(m, device=device)[: self.pca_block_size]
            weight_2d = weight_2d[row_idx]
            m = self.pca_block_size
        if n > self.pca_block_size:
            col_idx = torch.randperm(n, device=device)[: self.pca_block_size]
            weight_2d = weight_2d[:, col_idx]
            n = self.pca_block_size

        r        = min(self.pca_rank, m, n)
        validity = r / self.pca_rank

        try:
            U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
        except torch.linalg.LinAlgError:
            return torch.zeros(self.pca_rank * 3, device=device), 0.0

        S  = S[:r];  U  = U[:, :r];  Vh = Vh[:r, :]
        s_max  = S[0].clamp(min=1e-8)
        s_norm = S / s_max
        u_mean = U.mean(dim=0)
        v_mean = Vh.mean(dim=1)

        if r < self.pca_rank:
            pad_len = self.pca_rank - r
            s_norm = F.pad(s_norm, (0, pad_len))
            u_mean = F.pad(u_mean, (0, pad_len))
            v_mean = F.pad(v_mean, (0, pad_len))

        feature_vec = torch.cat([s_norm, u_mean, v_mean], dim=0)
        return feature_vec, validity

    @staticmethod
    def _compute_global_stats(weight_2d: torch.Tensor) -> torch.Tensor:
        """Compute five global statistics for a 2-D weight matrix.

        Args:
            weight_2d: 2-D float tensor.

        Returns:
            Tensor of shape (5,): [mean, std, abs_max, sparsity, rms].
        """
        flat     = weight_2d.flatten()
        mean     = flat.mean()
        std      = flat.std().clamp(min=1e-8)
        abs_max  = flat.abs().max()
        sparsity = (flat.abs() < 1e-6).float().mean()
        rms      = torch.linalg.norm(flat) / (flat.numel() ** 0.5)
        return torch.stack([mean, std, abs_max, sparsity, rms])

    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Encode weight_matrix through both branches and fuse.

        Args:
            weight_matrix: Arbitrary-shape weight tensor.

        Returns:
            Tensor of shape (emb_dim,).

        Time complexity:  O(N) for sparse branch + O(S^2) for PCA branch.
        Space complexity: O(emb_dim).
        """
        weight_f32  = weight_matrix.detach().float()
        weight_2d   = self._to_2d(weight_f32)
        weight_flat = weight_f32.flatten()

        sparse_feat, sparse_valid = self._compute_sparse_features(weight_flat)
        pca_feat,    pca_valid    = self._compute_pca_features(weight_2d)
        global_stats              = self._compute_global_stats(weight_2d)

        sparse_emb = self.sparse_mlp(sparse_feat) * sparse_valid
        pca_emb    = self.pca_mlp(pca_feat)       * pca_valid
        fused      = torch.cat([sparse_emb, pca_emb, global_stats], dim=0)
        return self.fusion_head(fused)


def build_weight_encoder(strategy: str) -> nn.Module:
    """Instantiate a weight encoder for the given strategy name.

    Args:
        strategy: One of "sparse", "pca", or "hybrid".

    Returns:
        An nn.Module encoder matching the strategy.

    Raises:
        ValueError: If strategy is not a known WeightEncoderStrategy.
    """
    strategy_enum = WeightEncoderStrategy(strategy.lower())
    if strategy_enum == WeightEncoderStrategy.SPARSE:
        return SparseWeightEncoder(
            top_k=Config.TOP_K_SPARSE_WEIGHTS,
            encoder_hidden=Config.WEIGHT_ENCODER_SPARSE_HIDDEN,
            emb_dim=Config.WEIGHT_EMB_DIM,
        )
    if strategy_enum == WeightEncoderStrategy.PCA:
        return PCAWeightEncoder(
            pca_rank=Config.PCA_RANK,
            pca_block_size=Config.PCA_BLOCK_SIZE,
            encoder_hidden=Config.WEIGHT_ENCODER_PCA_HIDDEN,
            emb_dim=Config.WEIGHT_EMB_DIM,
        )
    if strategy_enum == WeightEncoderStrategy.HYBRID:
        return HybridWeightEncoder(
            top_k=Config.TOP_K_SPARSE_WEIGHTS,
            pca_rank=Config.PCA_RANK,
            pca_block_size=Config.PCA_BLOCK_SIZE,
            sparse_hidden=Config.WEIGHT_ENCODER_SPARSE_HIDDEN,
            pca_hidden=Config.WEIGHT_ENCODER_PCA_HIDDEN,
            emb_dim=Config.WEIGHT_EMB_DIM,
            global_stats_dim=Config.GLOBAL_STATS_DIM,
        )
    raise ValueError(f"Unknown WeightEncoderStrategy: '{strategy}'.")


class WeightEmbeddingCache:
    """Pre-compute and cache weight embeddings for all teacher layers.

    The cache is built once at startup using the chosen encoder strategy and
    held in CPU memory as detached tensors.

    Args:
        teacher_model:      The teacher transformer model (must expose .layers).
        weight_names:       Dot-path attribute names to encode per layer.
        num_teacher_layers: Total number of teacher transformer layers.
        strategy:           Encoder strategy ("sparse", "pca", or "hybrid").
        device:             Device on which encoders run during construction.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        weight_names: List[str],
        num_teacher_layers: int,
        strategy: str,
        device: torch.device,
    ) -> None:
        self._emb_dim = Config.WEIGHT_EMB_DIM
        self._device  = device
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        logging.getLogger("RL-ILP-Agent").info(
            "Building WeightEmbeddingCache | strategy=%s | %d layers × %d matrices ...",
            strategy, num_teacher_layers, len(weight_names),
        )
        with torch.no_grad():
            for layer_idx in range(num_teacher_layers):
                self._cache[layer_idx] = {}
                teacher_layer = teacher_model.layers[layer_idx]
                for weight_name in weight_names:
                    weight_tensor = self._resolve_attribute(teacher_layer, weight_name)
                    if weight_tensor is None:
                        logging.getLogger("RL-ILP-Agent").warning(
                            "Layer %d: attribute '%s' not found — storing zeros.",
                            layer_idx, weight_name,
                        )
                        self._cache[layer_idx][weight_name] = torch.zeros(
                            self._emb_dim, device=device
                        )
                        continue
                    encoder = build_weight_encoder(strategy).to(device)
                    embedding = encoder(weight_tensor.to(device))
                    self._cache[layer_idx][weight_name] = embedding.detach()
                    del encoder

        logging.getLogger("RL-ILP-Agent").info(
            "WeightEmbeddingCache complete: %d entries, emb_dim=%d.",
            num_teacher_layers * len(weight_names), self._emb_dim,
        )

    def get(self, layer_idx: int, weight_name: str) -> torch.Tensor:
        """Retrieve the cached embedding for a specific layer and weight name.

        Args:
            layer_idx:   Teacher layer index.
            weight_name: Dot-path weight name (e.g. "self_attn.q_proj.weight").

        Returns:
            Tensor of shape (emb_dim,).
        """
        return self._cache[layer_idx][weight_name]

    def get_combined(self, layer_idx: int) -> torch.Tensor:
        """Return the mean of all weight embeddings for a given layer.

        Args:
            layer_idx: Teacher layer index.

        Returns:
            Tensor of shape (emb_dim,).
        """
        layer_embeddings = list(self._cache[layer_idx].values())
        stacked = torch.stack(layer_embeddings, dim=0)
        return stacked.mean(dim=0)

    @staticmethod
    def _resolve_attribute(
        module: nn.Module, dotted_path: str
    ) -> Optional[torch.Tensor]:
        """Walk a dotted attribute path on module, returning the leaf tensor.

        Args:
            module:      Root nn.Module to start from.
            dotted_path: Dot-separated path, e.g. "self_attn.q_proj.weight".

        Returns:
            The tensor at the leaf, or None if any part is missing.
        """
        obj = module
        for part in dotted_path.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj if isinstance(obj, torch.Tensor) else None


# ==============================================================================
# 3. STUDENT ARCHITECTURE
# ==============================================================================
class MultiHeadLatentAttention(nn.Module):
    """Multi-head attention with latent compression and rotary positional encoding.

    Implements MLA (Multi-head Latent Attention) from DeepSeek-v2:
    - Queries and keys/values are projected through a low-rank latent space.
    - RoPE is applied to a configurable sub-dimension of each head.
    - NoPE (no positional encoding) is used for the remaining sub-dimension.

    Args:
        hidden_dim: Model hidden dimension (H).
        num_heads:  Number of attention heads.
        latent_dim: Dimension of the latent compression space.
        rope_dim:   Number of dimensions per head that receive RoPE (must be
                    ≤ head_dim and even).
    """

    _LOGIT_CAP: float = 50.0

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        latent_dim: int,
        rope_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        if hidden_dim % num_heads != 0:
            logging.getLogger("RL-ILP-Agent").warning(
                "hidden_dim (%d) not divisible by num_heads (%d); "
                "head_dim truncated to %d.",
                hidden_dim, num_heads, self.head_dim,
            )
        self.attn_out_dim = self.num_heads * self.head_dim
        if rope_dim > self.head_dim:
            raise ValueError(
                f"rope_dim ({rope_dim}) must be ≤ head_dim ({self.head_dim})."
            )
        self.rope_dim  = rope_dim
        self.nope_dim  = self.head_dim - rope_dim
        self._rope_scale = 1.0

        self.q_down      = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.q_up        = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        self.kv_down     = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.kv_up       = nn.Linear(
            latent_dim, num_heads * (self.nope_dim + self.head_dim), bias=False
        )
        self.k_up_rope   = nn.Linear(hidden_dim, num_heads * rope_dim, bias=False)
        self.q_norm      = nn.RMSNorm(self.head_dim)
        self.k_norm      = nn.RMSNorm(self.head_dim)
        self.out_proj    = nn.Linear(self.attn_out_dim, hidden_dim, bias=False)
        self._scale      = 1.0 / math.sqrt(self.head_dim)

        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
        self._cached_max_pos = 0
        self._cached_scale   = 1.0

    def set_rope_scale(self, scale: float) -> None:
        """Set the NTK RoPE scaling factor and invalidate the cache.

        Args:
            scale: New scale factor (> 1.0 extends the effective context window).
        """
        if scale != self._rope_scale:
            self._rope_scale    = scale
            self._cached_max_pos = 0

    @staticmethod
    def create_padding_mask(
        seq_lens: List[int], max_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build a boolean padding mask from a list of sequence lengths.

        Args:
            seq_lens: Per-sequence valid token counts.
            max_len:  Padded sequence length.
            device:   Target device.

        Returns:
            Bool tensor of shape (B, max_len); True where tokens are valid.
        """
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        lengths   = torch.tensor(seq_lens, device=device).unsqueeze(1)
        return positions < lengths

    def _update_rope_cache(self, max_pos: int, device: torch.device) -> None:
        """Rebuild the cosine/sine RoPE cache when needed.

        Args:
            max_pos: Maximum position index required.
            device:  Compute device.
        """
        if max_pos <= self._cached_max_pos and self._rope_scale == self._cached_scale:
            return

        target_len = max(max_pos + 128, self._cached_max_pos * 2, 1024)
        half_d     = self.rope_dim // 2
        freq_seq   = torch.arange(half_d, device=device, dtype=torch.float32)
        inv_freq   = 1.0 / (
            (10_000.0 * self._rope_scale) ** (freq_seq * 2.0 / self.rope_dim)
        )
        positions  = torch.arange(target_len, device=device, dtype=torch.float32)
        angles     = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos_raw, sin_raw = torch.cos(angles), torch.sin(angles)

        even_dim = half_d * 2
        cos_full = torch.zeros(target_len, even_dim, device=device)
        sin_full = torch.zeros(target_len, even_dim, device=device)
        cos_full[:, 0::2] = cos_raw;  cos_full[:, 1::2] = cos_raw
        sin_full[:, 0::2] = sin_raw;  sin_full[:, 1::2] = sin_raw

        self._cos_cached = cos_full.unsqueeze(0).unsqueeze(2)
        self._sin_cached = sin_full.unsqueeze(0).unsqueeze(2)
        self._cached_max_pos = target_len
        self._cached_scale   = self._rope_scale

    def _get_rope(
        self, pos_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather RoPE cos/sin tensors for the given position IDs.

        Args:
            pos_ids: Integer tensor of shape (B, S).

        Returns:
            Tuple (cos, sin), each of shape (B, S, 1, rope_dim).
        """
        max_pos = int(pos_ids.max().item())
        self._update_rope_cache(max_pos + 1, pos_ids.device)
        flat_ids     = pos_ids.flatten()
        cos_gathered = self._cos_cached.squeeze(0).index_select(0, flat_ids)
        sin_gathered = self._sin_cached.squeeze(0).index_select(0, flat_ids)
        B, S         = pos_ids.shape
        cos          = cos_gathered.view(B, S, 1, -1)
        sin          = sin_gathered.view(B, S, 1, -1)
        return cos, sin

    def _apply_rope(
        self,
        token_tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position encoding to token_tensor.

        Args:
            token_tensor: Query or key tensor slice for RoPE dimensions.
            cos:          Cosine values, shape (B, S, 1, rope_dim).
            sin:          Sine values,   shape (B, S, 1, rope_dim).

        Returns:
            Rotated tensor of the same shape as token_tensor.
        """
        rope_dim = self.rope_dim
        if rope_dim % 2 == 1:
            t_rot  = token_tensor[..., : rope_dim - 1]
            t_pass = token_tensor[..., rope_dim - 1:]
        else:
            t_rot  = token_tensor
            t_pass = None

        t_rotated = torch.stack(
            [-t_rot[..., 1::2], t_rot[..., 0::2]], dim=-1
        ).flatten(-2)
        t_out = t_rot * cos + t_rotated * sin

        if t_pass is not None:
            t_out = torch.cat([t_out, t_pass], dim=-1)
        return t_out

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute multi-head latent attention with RoPE.

        Args:
            x:               Input tensor (B, S, H).
            pos_ids:         Position indices (B, S).
            mask:            Optional attention mask.
            past_key_value:  Optional KV cache for incremental decoding.
            use_cache:       If True, return updated KV cache.

        Returns:
            Tuple of (output tensor (B, S, H), optional updated KV cache).

        Time complexity:  O(B * S^2 * H) for attention computation.
        Space complexity: O(B * num_heads * S * head_dim) for KV storage.
        """
        B, S, _ = x.shape

        q_lat  = self.q_down(x)
        q_all  = self.q_up(q_lat).view(B, S, self.num_heads, self.head_dim)
        kv_lat = self.kv_down(x)
        kv_all = self.kv_up(kv_lat).view(
            B, S, self.num_heads, self.nope_dim + self.head_dim
        )
        k_nope     = kv_all[..., : self.nope_dim]
        v          = kv_all[..., self.nope_dim:]
        k_rope_src = self.k_up_rope(x).view(B, S, self.num_heads, self.rope_dim)
        k_all      = torch.cat([k_nope, k_rope_src], dim=-1)

        orig_dtype = q_all.dtype
        q = self.q_norm(q_all.float()).to(orig_dtype)
        k = self.k_norm(k_all.float()).to(orig_dtype)

        q_nope = q[..., : self.nope_dim];  q_rope = q[..., self.nope_dim:]
        k_nope = k[..., : self.nope_dim];  k_rope = k[..., self.nope_dim:]

        cos, sin = self._get_rope(pos_ids)
        q_rope   = self._apply_rope(q_rope, cos, sin)
        k_rope   = self._apply_rope(k_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k, v) if use_cache else None

        attn_mask = None
        is_causal = False
        if mask is None:
            is_causal = S > 1
        elif mask.ndim == 2 and mask.shape == (B, S) and mask.dtype == torch.bool:
            causal_mask = torch.ones((S, S), device=x.device, dtype=torch.bool).triu(1)
            padding_expanded = (~mask).view(B, 1, 1, S)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) | padding_expanded
            is_causal = False
        else:
            attn_mask = mask
            is_causal = False

        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self._scale,
        )
        attended = attended.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(attended), current_kv


class ExpertFFN(nn.Module):
    """Gated feed-forward network for a single MoE expert.

    Uses SiLU gating (SwiGLU variant) with an optional depth-aware output scale
    applied to the down-projection at initialisation.

    Args:
        hidden_dim: Input and output dimension.
        inter_dim:  Intermediate (gated) dimension.
        num_layers: Model depth for depth-aware weight scaling.
    """

    def __init__(
        self, hidden_dim: int, inter_dim: int, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(hidden_dim, inter_dim * 2, bias=False)
        self.w3        = nn.Linear(inter_dim, hidden_dim, bias=False)
        self._inter_dim = inter_dim
        depth_scale = 1.0 / math.sqrt(2.0 * max(num_layers, 1))
        self.register_buffer(
            "_depth_scale",
            torch.tensor(depth_scale, dtype=torch.float32),
            persistent=False,
        )
        with torch.no_grad():
            self.w3.weight.mul_(depth_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated FFN transformation.

        Args:
            x: Input tensor (..., hidden_dim).

        Returns:
            Output tensor (..., hidden_dim).

        Time complexity:  O(B * S * hidden_dim * inter_dim).
        Space complexity: O(B * S * inter_dim).
        """
        fused = self.w_gate_up(x)
        gate  = fused[..., : self._inter_dim]
        up    = fused[..., self._inter_dim:]
        return self.w3(F.silu(gate, inplace=True) * up)


class WeightAwareMoELayer(nn.Module):
    """Mixture-of-Experts layer with weight-embedding-conditioned routing.

    The router receives a concatenation of the token hidden state and the
    teacher weight embedding so that expert selection is conditioned on the
    structural characteristics of the corresponding teacher layer.

    Args:
        hidden_dim:        Token hidden dimension.
        inter_dim:         Expert FFN intermediate dimension.
        num_experts:       Total number of experts.
        active_experts:    Top-K experts selected per token (K).
        weight_emb_dim:    Dimension of the injected weight embedding.
        gate_dropout:      Dropout applied to router inputs during training.
        gate_temperature:  Softmax temperature for the router.
        noisy_gating:      Add learned noise for load-balance exploration.
        noise_epsilon:     Minimum noise std (numerical stability).
        capacity_factor:   Optional per-expert token capacity ratio.
        entropy_coeff:     Weight of the negative gate-entropy auxiliary loss.
        load_balance_coeff: Weight of the load-balance auxiliary loss.
        eps:               Numerical epsilon.
        use_residual:      Whether to apply a residual inside the MoE block.
                           Set False when the caller owns the residual.
    """

    # ── Inner Router ────────────────────────────────────────────────────────
    class Router(nn.Module):
        """Noisy-top-K softmax router for MoE expert selection.

        Args:
            input_dim:        Dimension of router input (hidden + weight_emb).
            num_experts:      Total expert count.
            k:                Experts activated per token.
            temperature:      Softmax temperature.
            noisy_gating:     Add multiplicative noise during training.
            noise_epsilon:    Minimum noise std.
            dropout:          Input dropout probability.
            capacity_factor:  Reserved; not enforced in this implementation.
            eps:              Numerical epsilon.
        """

        def __init__(
            self,
            input_dim: int,
            num_experts: int,
            k: int,
            temperature: float = 0.875,
            noisy_gating: bool = True,
            noise_epsilon: float = 1e-2,
            dropout: float = 0.0,
            capacity_factor: Optional[float] = None,
            eps: float = 1e-9,
        ) -> None:
            super().__init__()
            assert 1 <= k <= num_experts
            self.num_experts     = num_experts
            self.k               = k
            self.temperature     = temperature
            self.noisy_gating    = noisy_gating
            self.noise_epsilon   = noise_epsilon
            self.capacity_factor = capacity_factor
            self.eps             = eps
            self.gate    = nn.Linear(input_dim, num_experts, bias=True)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            if noisy_gating:
                self.w_noise  = nn.Linear(input_dim, num_experts, bias=False)
                self.softplus = nn.Softplus()
                self.register_buffer("mean", torch.tensor([0.0]))
                self.register_buffer("std",  torch.tensor([1.0]))

        def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
            """Count how many tokens each expert receives.

            Args:
                gates: Gate matrix (num_tokens, num_experts).

            Returns:
                Expert load counts (num_experts,).
            """
            return (gates > 0).sum(0)

        def _prob_in_top_k(
            self,
            clean_logits: torch.Tensor,
            noisy_logits: torch.Tensor,
            noise_stddev: torch.Tensor,
            noisy_top_logits: torch.Tensor,
        ) -> torch.Tensor:
            """Estimate the probability that each expert is in the top-K.

            Args:
                clean_logits:     Pre-noise gate logits (B, num_experts).
                noisy_logits:     Post-noise gate logits (B, num_experts).
                noise_stddev:     Per-expert noise std (B, num_experts).
                noisy_top_logits: Top-(K+1) noisy logits (B, K+1).

            Returns:
                Probability tensor (B, num_experts).
            """
            batch, _ = clean_logits.shape
            m = noisy_top_logits.size(1)
            top_flat = noisy_top_logits.flatten()

            threshold_pos = (
                torch.arange(batch, device=clean_logits.device) * m + self.k
            )
            threshold = torch.gather(top_flat, 0, threshold_pos).unsqueeze(1)
            is_in = noisy_logits > threshold

            threshold_pos_out = threshold_pos - 1
            threshold_out = torch.gather(top_flat, 0, threshold_pos_out).unsqueeze(1)

            normal   = Normal(self.mean, self.std)
            prob_in  = normal.cdf((clean_logits - threshold)     / noise_stddev)
            prob_out = normal.cdf((clean_logits - threshold_out) / noise_stddev)
            return torch.where(is_in, prob_in, prob_out)

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute router scores, top-K expert indices, and aux loss.

            Args:
                x: Router input (num_tokens, input_dim).

            Returns:
                Tuple of (scores, topk_idx, topk_vals, aux_loss).

            Time complexity:  O(num_tokens * input_dim * num_experts).
            Space complexity: O(num_tokens * num_experts).
            """
            x            = self.dropout(x)
            clean_logits = self.gate(x)

            if self.noisy_gating and self.training:
                raw_noise_stddev = self.w_noise(x)
                noise_stddev     = self.softplus(raw_noise_stddev) + self.noise_epsilon
                logits           = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            else:
                logits       = clean_logits
                noise_stddev = None

            logits = logits / (self.temperature + self.eps)
            scores = F.softmax(logits, dim=-1)
            topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)

            importance    = scores.sum(dim=0)
            dispatch_mask = torch.zeros_like(scores)
            dispatch_mask.scatter_(1, topk_idx, 1.0)
            load = dispatch_mask.sum(dim=0)

            num_tokens = x.size(0)
            aux_loss   = (importance * load).sum() * (
                self.num_experts / (num_tokens ** 2 + self.eps)
            )
            return scores, topk_idx, topk_vals, aux_loss

    # ── Inner SparseDispatcher ───────────────────────────────────────────────
    class SparseDispatcher:
        """Efficiently dispatch tokens to experts and combine results.

        Args:
            num_experts: Total expert count.
            gates:       Full gate matrix (num_tokens, num_experts).
            topk_idx:    Top-K expert indices (num_tokens, k).
            topk_vals:   Top-K renormalised gate values (num_tokens, k).
                         Must be the RENORMALISED values.
        """

        def __init__(
            self,
            num_experts: int,
            gates: torch.Tensor,
            topk_idx: torch.Tensor,
            topk_vals: torch.Tensor,
        ) -> None:
            self._gates      = gates
            self._num_experts = num_experts
            self._topk_idx   = topk_idx
            self._topk_vals  = topk_vals

            num_tokens = gates.size(0)
            k          = topk_idx.size(1)

            expert_indices = topk_idx.flatten()
            token_indices  = torch.arange(
                num_tokens, device=gates.device
            ).repeat_interleave(k)
            values         = topk_vals.flatten()

            sorted_order = expert_indices.argsort(stable=True)
            self._expert_indices_sorted = expert_indices[sorted_order]
            self._token_indices_sorted  = token_indices[sorted_order]
            self._values_sorted         = values[sorted_order]

            expert_limits = (
                self._expert_indices_sorted
                == torch.arange(num_experts, device=gates.device).unsqueeze(1)
            ).sum(dim=1)
            self._part_sizes          = expert_limits.tolist()
            self._split_token_indices = torch.split(
                self._token_indices_sorted, self._part_sizes
            )
            self._split_values = torch.split(
                self._values_sorted, self._part_sizes
            )

        def dispatch(self, inp: torch.Tensor) -> List[torch.Tensor]:
            """Gather token vectors for each expert.

            Args:
                inp: Input tensor (num_tokens, hidden_dim).

            Returns:
                List of per-expert input tensors.
            """
            inp_exp = inp[self._token_indices_sorted]
            return list(torch.split(inp_exp, self._part_sizes, dim=0))

        def combine(
            self,
            expert_outputs: List[torch.Tensor],
            multiply_by_gates: bool = True,
        ) -> torch.Tensor:
            """Scatter-add expert outputs back to token positions.

            Args:
                expert_outputs:    Per-expert output tensors.
                multiply_by_gates: Whether to weight by gate values.

            Returns:
                Combined tensor (num_tokens, hidden_dim).
            """
            stitched = torch.cat(expert_outputs, dim=0)
            if multiply_by_gates:
                stitched = stitched * self._values_sorted.unsqueeze(-1)
            combined = torch.zeros(
                self._gates.size(0),
                expert_outputs[-1].size(1),
                device=stitched.device,
                dtype=stitched.dtype,
            )
            combined.index_add_(0, self._token_indices_sorted, stitched)
            return combined

        def expert_to_gates(self) -> List[torch.Tensor]:
            """Return per-expert gate value slices.

            Returns:
                List of gate value tensors, one per expert.
            """
            return list(self._split_values)

    # ── WeightAwareMoELayer init & forward ───────────────────────────────────
    def __init__(
        self,
        hidden_dim: int,
        inter_dim: int,
        num_experts: int,
        active_experts: int,
        weight_emb_dim: int,
        gate_dropout: float = 0.1,
        gate_temperature: float = 0.875,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2,
        capacity_factor: Optional[float] = None,
        entropy_coeff: float = 0.01,
        load_balance_coeff: float = MOE_LOAD_BALANCE_COEFF,
        eps: float = 1e-9,
        use_residual: bool = True,
        # Soft routing: when True, all experts are active with weighted outputs
        soft_routing: bool = True,
        gate_noise_std: float = MOE_GATE_NOISE_STD,
    ) -> None:
        super().__init__()
        assert 1 <= active_experts <= num_experts
        self.num_experts    = num_experts
        self.active_experts = active_experts
        self.hidden_dim     = hidden_dim
        self.weight_emb_dim = weight_emb_dim
        self.use_residual   = use_residual
        self._eps           = eps
        # Store routing configuration
        self.soft_routing   = soft_routing
        self.gate_noise_std = gate_noise_std

        self.router = self.Router(
            input_dim=hidden_dim + weight_emb_dim,
            num_experts=num_experts,
            k=active_experts,
            temperature=gate_temperature,
            noisy_gating=noisy_gating,
            noise_epsilon=noise_epsilon,
            dropout=gate_dropout,
            capacity_factor=capacity_factor,
            eps=eps,
        )
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_dim, inter_dim) for _ in range(num_experts)
        ])
        self.shared_expert = ExpertFFN(hidden_dim, inter_dim)
        self.shared_gate   = nn.Linear(hidden_dim, 1)
        self.combine_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.entropy_coeff      = entropy_coeff
        self.load_balance_coeff = load_balance_coeff
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _flatten_batch(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Reshape (B, S, H) → (B*S, H) and return the original (B, S).

        Args:
            x: Input tensor of shape (B, S, H).

        Returns:
            Tuple of (flattened tensor, (B, S)).
        """
        B, S, H = x.shape
        return x.view(B * S, H), (B, S)

    @staticmethod
    def _unflatten_batch(
        x_flat: torch.Tensor, shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Reshape (B*S, H) → (B, S, H).

        Args:
            x_flat: Flattened tensor (B*S, H).
            shape:  Original (B, S).

        Returns:
            Tensor of shape (B, S, H).
        """
        B, S = shape
        return x_flat.view(B, S, -1)

    def _compute_gate_entropy(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute the mean entropy of the gate distribution.

        Args:
            gates: Normalised gate matrix (num_tokens, num_experts).

        Returns:
            Scalar entropy tensor.
        """
        p    = gates.clamp(min=self._eps)
        logp = torch.log(p)
        return -(p * logp).sum(dim=-1).mean()

    def forward(
        self, x: torch.Tensor, weight_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens through experts conditioned on the weight embedding.

        When soft_routing=True the output is a gate-weighted sum of ALL experts'
        outputs (no sparse dispatch), preventing binary routing lock-in that kills
        2-expert models during RL. Gate-logit noise (σ=gate_noise_std) further
        prevents router collapse.

        Args:
            x:          Input tensor (B, S, H).
            weight_emb: Weight embedding; accepted shapes:
                        - (W,)      — broadcast over batch and sequence.
                        - (B, W)    — broadcast over sequence.
                        - (B, S, W) — per-token embedding (averaged over S).

        Returns:
            Tuple of (output tensor (B, S, H), scalar auxiliary loss).

        Time complexity:  O(B * S * num_experts * H * inter_dim) — soft routing.
        Space complexity: O(B * S * num_experts).
        """
        B, S, H = x.shape
        device  = x.device

        if weight_emb.dim() == 1:
            weight_emb = weight_emb.unsqueeze(0).expand(B, -1)
        elif weight_emb.dim() == 3 and weight_emb.shape[:2] == (B, S):
            weight_emb = weight_emb.mean(dim=1)
        elif weight_emb.dim() == 2 and weight_emb.shape[0] == B:
            pass  # already (B, W)
        else:
            raise ValueError(
                "weight_emb must be shape [W], [B, W], or [B, S, W]."
            )

        x_flat, batch_shape = self._flatten_batch(x)
        BS    = x_flat.shape[0]
        w_exp = weight_emb.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)

        router_input = torch.cat([x_flat, w_exp], dim=-1)

        # Add gate-logit noise to prevent routing lock-in
        gate_logits = self.router.gate(router_input)
        if self.training and self.gate_noise_std > 0.0:
            gate_logits = gate_logits + self.gate_noise_std * torch.randn_like(gate_logits)

        if self.soft_routing:
            # Soft routing: all experts always active
            gate_probs = F.softmax(
                gate_logits / (self.router.temperature + self.router.eps), dim=-1
            )

            importance = gate_probs.sum(dim=0)
            load_term  = self.load_balance_coeff * (
                self.num_experts / (BS ** 2 + self.router.eps)
            ) * (importance * importance).sum()

            expert_outs = torch.stack(
                [expert(x_flat) for expert in self.experts], dim=1
            )  # (BS, num_experts, H)
            fused_flat = (gate_probs.unsqueeze(-1) * expert_outs).sum(dim=1)  # (BS, H)
            router_aux = load_term
            gates      = gate_probs
        else:
            # ── Sparse top-K dispatch (legacy path) ───────────────────────────
            scores, topk_idx, topk_vals, router_aux = self.router(router_input)
            gates     = torch.zeros_like(scores)
            row_idx   = torch.arange(BS, device=device).unsqueeze(1).expand(
                -1, self.active_experts
            )
            gates[row_idx, topk_idx] = topk_vals
            gates_sum = gates.sum(dim=-1, keepdim=True).clamp(min=self._eps)
            gates     = gates / gates_sum

            # Use renormalised values in the dispatcher.
            renorm_topk_vals = gates[row_idx, topk_idx]
            dispatcher       = self.SparseDispatcher(
                self.num_experts, gates, topk_idx, renorm_topk_vals
            )
            expert_inputs  = dispatcher.dispatch(x_flat)
            expert_outputs = [
                expert(expert_inputs[i])
                if expert_inputs[i].size(0) > 0
                else torch.zeros(0, self.hidden_dim, device=device)
                for i, expert in enumerate(self.experts)
            ]
            fused_flat = dispatcher.combine(expert_outputs, multiply_by_gates=True)

        # Shared expert blending (applies to both routing modes).
        shared_weight = torch.sigmoid(self.shared_gate(x_flat))
        shared_out    = self.shared_expert(x_flat)
        fused_flat    = fused_flat * (1.0 - shared_weight) + shared_out * shared_weight
        fused_flat    = self.combine_proj(fused_flat)

        moe_out = self._unflatten_batch(fused_flat, batch_shape)
        out     = x + moe_out if self.use_residual else moe_out

        gates_det       = gates.detach()
        expert_loads    = gates_det.mean(dim=0)
        gate_entropy    = self._compute_gate_entropy(gates)
        entropy_term    = -self.entropy_coeff * gate_entropy
        aux_loss        = (router_aux + entropy_term).to(out.dtype)

        self.last_diagnostics = {
            "gates":             gates_det,
            "expert_loads":      expert_loads,
            "gate_importance":   gates_det.mean(dim=0),
            "load_balance_loss": router_aux.detach(),
            "gate_entropy":      gate_entropy.detach(),
            "router_entropy_bits": (gate_entropy / math.log(2)).detach(),
        }
        return out, aux_loss


class WeightAwareDifferentiableMemory(nn.Module):
    """Differentiable key-value memory conditioned on teacher weight embeddings.

    The memory is queried with normalised student hidden states and the
    result is gated by a learnable function of the query, retrieved values,
    and the weight embedding.  Only the DELTA (gated projection) is returned;
    the caller is responsible for the residual connection.

    Args:
        num_slots:      Number of memory slots.
        key_dim:        Key and output dimension (must equal student hidden_dim).
        val_dim:        Value dimension stored in memory.
        top_k:          Number of slots retrieved per query.
        teacher_dim:    Teacher hidden dimension for alignment projection.
        weight_emb_dim: Weight embedding dimension for gate conditioning.
    """

    def __init__(
        self,
        num_slots: int,
        key_dim: int,
        val_dim: int,
        top_k: int,
        teacher_dim: int,
        weight_emb_dim: int,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.key_dim   = key_dim
        self.val_dim   = val_dim
        self.top_k     = top_k

        self.keys   = nn.Parameter(torch.empty(num_slots, key_dim))
        self.values = nn.Parameter(torch.empty(num_slots, val_dim))

        self.query_proj = nn.Linear(key_dim, key_dim, bias=False)
        self.input_norm = nn.RMSNorm(key_dim)
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, key_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(key_dim * 2, key_dim, bias=False),
        )
        gate_input_dim = key_dim + val_dim + weight_emb_dim
        self.out_gate = nn.Sequential(
            nn.Linear(gate_input_dim, key_dim, bias=False),
            nn.LayerNorm(key_dim),
            nn.Tanh(),
            nn.Linear(key_dim, key_dim, bias=True),
        )
        self.out_proj = nn.Linear(val_dim, key_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Orthogonal init for all linear layers; normal init for memory values."""
        nn.init.orthogonal_(self.keys,  gain=1.0)
        nn.init.normal_(self.values, std=0.02)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)
        for module in self.teacher_proj:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        for module in self.out_gate:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        student_h: torch.Tensor,
        weight_emb: torch.Tensor,
        teacher_h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query memory and return a gated delta for residual addition.

        Args:
            student_h:  Student hidden states (B, S, key_dim).
            weight_emb: Weight embedding (weight_emb_dim,).
            teacher_h:  Optional teacher hidden states for alignment (B, S, teacher_dim).

        Returns:
            Tuple of:
              - delta:      Gated memory delta (B, S, key_dim).  Add to residual externally.
              - align_loss: MSE alignment loss (scalar); 0 if teacher_h is None.

        Time complexity:  O(B * S * num_slots) for attention over memory.
        Space complexity: O(B * S * top_k * val_dim).
        """
        B, S, _ = student_h.shape
        x_norm  = self.input_norm(student_h)
        q       = self.query_proj(x_norm)
        q_norm  = F.normalize(q, dim=-1)
        k_norm  = F.normalize(self.keys, dim=-1)
        scores  = torch.matmul(q_norm, k_norm.t())

        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_scores, dim=-1).unsqueeze(-1)

        flat_indices    = topk_indices.view(-1)
        retrieved_vals  = self.values[flat_indices].view(B, S, self.top_k, self.val_dim)
        read_v          = (topk_probs * retrieved_vals).sum(dim=2)

        weight_emb_expanded = weight_emb.detach().unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        gate_input = torch.cat([q, read_v, weight_emb_expanded], dim=-1)
        gate       = torch.sigmoid(self.out_gate(gate_input))
        projected_v = self.out_proj(read_v)

        # Return DELTA only, scaled to prevent memory from dominating residual stream
        delta = gate * projected_v * MEMORY_OUTPUT_SCALE

        align_loss = torch.zeros(1, device=student_h.device).squeeze()
        if teacher_h is not None:
            t_concept = F.normalize(
                self.teacher_proj(teacher_h.detach()), dim=-1
            )
            retrieved_keys = self.keys[flat_indices].view(
                B, S, self.top_k, self.key_dim
            )
            retrieved_keys   = F.normalize(retrieved_keys, dim=-1)
            student_concept  = (topk_probs * retrieved_keys).sum(dim=2)
            align_loss       = F.mse_loss(student_concept, t_concept)

        return delta, align_loss


# ==============================================================================
# 4. MODELS
# ==============================================================================
class DeepSeekTeacher(nn.Module):
    """Frozen DeepSeek teacher model wrapper.

    Loads the HuggingFace checkpoint, exposes its tokenizer, and provides
    forward / generate methods.  All parameters are frozen after loading.

    Args:
        None — architecture is controlled by Config.TEACHER_ARCH.
    """

    def __init__(self) -> None:
        super().__init__()
        model_id = Config.get_teacher_model_id()
        logging.getLogger("RL-ILP-Agent").info(
            "Loading Teacher: %s  (arch=%s) ...", model_id, Config.TEACHER_ARCH
        )
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                trust_remote_code=True,
            )
            self.tokenizer  = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model      = hf_model
            self.transformer: nn.Module = hf_model.model
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        except OSError as exc:
            logging.getLogger("RL-ILP-Agent").error(
                "Failed to load Teacher model: %s", exc
            )
            raise

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> Dict[str, object]:
        """Run teacher forward pass and return logits + hidden states.

        Args:
            input_ids: Integer token tensor (B, S).

        Returns:
            Dict with keys "logits", "hidden_states", "attentions".
        """
        outputs = self.model(
            input_ids, output_hidden_states=True, output_attentions=True
        )
        return {
            "logits":        outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions":    outputs.attentions,
        }

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Delegate to the HuggingFace model.generate with no-grad context.

        Returns:
            Generated token IDs tensor.
        """
        return self.model.generate(*args, **kwargs)


class DeepMimeStudent(nn.Module):
    """Small student model that mimics the DeepSeek teacher.

    Architecture:
        - Token embedding + tied LM head.
        - N layers, each containing: MLA attention, differentiable memory,
          weight-aware MoE FFN.
        - Final RMSNorm before the LM head.

    Args:
        None — all dimensions are read from Config.
    """

    _MAX_BUFFER_LEN: int = 8192

    def __init__(self) -> None:
        super().__init__()
        vocab_size   = Config.get_vocab_size()
        teacher_hdim = Config.get_teacher_hidden_dim()

        self.embed = nn.Embedding(vocab_size, Config.HIDDEN_DIM)
        # Register causal buffer for fallback masking
        causal_buf = torch.ones(
            self._MAX_BUFFER_LEN, self._MAX_BUFFER_LEN, dtype=torch.bool
        ).triu(diagonal=1)
        self.register_buffer("_causal_buf", causal_buf, persistent=False)

        self._weight_cache: Optional[WeightEmbeddingCache] = None
        self._teacher_stride: int = (
            Config.get_teacher_num_layers() // Config.NUM_LAYERS
        )
        self._use_grad_ckpt: bool = False

        self.layers      = nn.ModuleList()
        self.align_projs = nn.ModuleList()

        for _ in range(Config.NUM_LAYERS):
            self.layers.append(
                nn.ModuleDict({
                    "norm1": nn.RMSNorm(Config.HIDDEN_DIM),
                    "attn":  MultiHeadLatentAttention(
                        Config.HIDDEN_DIM, Config.NUM_HEADS,
                        Config.LATENT_DIM, Config.ROPE_DIM,
                    ),
                    "norm2": nn.RMSNorm(Config.HIDDEN_DIM),
                    "mem":   WeightAwareDifferentiableMemory(
                        num_slots=Config.MEMORY_SLOTS,
                        key_dim=Config.HIDDEN_DIM,
                        val_dim=Config.VALUE_DIM,
                        top_k=Config.MEMORY_TOP_K,
                        teacher_dim=teacher_hdim,
                        weight_emb_dim=Config.WEIGHT_EMB_DIM,
                    ),
                    "norm3": nn.RMSNorm(Config.HIDDEN_DIM),
                    "moe":   WeightAwareMoELayer(
                        hidden_dim=Config.HIDDEN_DIM,
                        inter_dim=Config.INTERMEDIATE_DIM,
                        num_experts=Config.NUM_EXPERTS,
                        active_experts=Config.ACTIVE_EXPERTS,
                        weight_emb_dim=Config.WEIGHT_EMB_DIM,
                        use_residual=False, # Caller handles residual
                    ),
                })
            )
            self.align_projs.append(
                nn.Linear(Config.HIDDEN_DIM, teacher_hdim)
            )

        self.norm_final = nn.RMSNorm(Config.HIDDEN_DIM)
        self.lm_head    = nn.Linear(Config.HIDDEN_DIM, vocab_size, bias=False)

        # Initialise weights before tying
        self.apply(self._init_weights)
        self._apply_output_scaling()
        
        # Tie weights: embed and lm_head share the same parameters
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Orthogonal init for Linear layers; normal init for Embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_output_scaling(self) -> None:
        """Scale attention and MoE output projections by 1/sqrt(2*NUM_LAYERS)."""
        scale = 1.0 / math.sqrt(2.0 * Config.NUM_LAYERS)
        for layer in self.layers:
            if hasattr(layer["attn"], "out_proj"):
                with torch.no_grad():
                    layer["attn"].out_proj.weight.mul_(scale)
            if hasattr(layer["moe"], "combine_proj"): # Check correct attr name
                with torch.no_grad():
                    layer["moe"].combine_proj.weight.mul_(scale)

    def enable_gradient_checkpointing(self, value: bool = True) -> None:
        self._use_grad_ckpt = value

    def set_weight_cache(self, cache: WeightEmbeddingCache) -> None:
        self._weight_cache = cache

    def _layer_fn(
        self,
        x: torch.Tensor,
        layer: nn.ModuleDict,
        layer_weight_emb: torch.Tensor,
        pos_ids: torch.Tensor,
        mask: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
        teacher_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # 1. Attention Block
        residual = x
        attn_out, current_kv = layer["attn"](
            layer["norm1"](x), pos_ids, mask=mask,
            past_key_value=layer_past, use_cache=use_cache,
        )
        x = residual + attn_out

        # 2. Memory Block
        residual = x
        mem_delta, mem_loss = layer["mem"](
            layer["norm2"](x), weight_emb=layer_weight_emb, teacher_h=teacher_state
        )
        x = residual + mem_delta

        # 3. MoE Block
        residual = x
        moe_out, lb_loss = layer["moe"](layer["norm3"](x), weight_emb=layer_weight_emb)
        x = residual + moe_out

        return x, mem_loss + lb_loss, current_kv

    def forward(
        self,
        input_ids: torch.Tensor,
        teacher_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, object]:
        """Full forward pass of the student model."""
        if self._weight_cache is None:
            raise RuntimeError("WeightEmbeddingCache has not been set.")

        B, S = input_ids.shape
        
        # CRITICAL FIX: Scale embeddings by sqrt(dim).
        # Without this, input variance is ~0.0004 vs residual variance ~1.0,
        # causing the model to completely ignore the input tokens ("ctctctct...").
        x = self.embed(input_ids) * math.sqrt(Config.HIDDEN_DIM)

        if past_key_values is None:
            past_length = 0
        else:
            first_kv = past_key_values[0]
            pk = first_kv[0] if isinstance(first_kv, (tuple, list)) else first_kv
            past_length = pk.shape[2] if pk.dim() == 4 else pk.shape[1]

        pos_ids = torch.arange(
            past_length, past_length + S, device=x.device
        ).unsqueeze(0).expand(B, -1)

        # Simplified mask logic: 
        # If we are training (S > 1, no past), use causal mask.
        # If we are generating (S == 1, past exists), is_causal=False handles it internally.
        mask = None
        if past_key_values is None and S > 1:
            # Create a boolean mask where True indicates "do not attend" (future positions)
            # _causal_buf is 1 in upper triangle (future).
            # SDPA expects:
            # - is_causal=True (handles it efficiently)
            # - OR attn_mask where True = keep, False = mask (if float)
            # - OR attn_mask where True = mask, False = keep (if bool) -> PyTorch convention varies!
            # Safest is to let SDPA handle causal masking via is_causal=True when mask is None.
            pass 

        total_len = past_length + S
        base_len  = getattr(Config, "MAX_SEQ_LEN", 2048)
        if total_len > base_len:
            ntk_scale = (total_len / base_len) ** (
                Config.ROPE_DIM / max(Config.ROPE_DIM - 2, 1)
            )
            for layer in self.layers:
                if hasattr(layer["attn"], "set_rope_scale"):
                    layer["attn"].set_rope_scale(ntk_scale)

        aux_losses:            List[torch.Tensor] = []
        student_hidden_states: List[torch.Tensor] = []
        new_key_values: Optional[List] = [] if use_cache else None

        for student_idx, layer in enumerate(self.layers):
            teacher_idx      = student_idx * self._teacher_stride
            layer_weight_emb = self._weight_cache.get_combined(teacher_idx)
            layer_past       = (
                past_key_values[student_idx] if past_key_values is not None else None
            )
            teacher_state = (
                teacher_hidden_states[teacher_idx]
                if (teacher_hidden_states is not None and not use_cache)
                else None
            )
            
            if self._use_grad_ckpt and self.training:
                # Gradient checkpointing logic would go here
                x, layer_aux, current_kv = self._layer_fn(
                    x, layer, layer_weight_emb, pos_ids,
                    mask, layer_past, use_cache, teacher_state
                )
            else:
                x, layer_aux, current_kv = self._layer_fn(
                    x, layer, layer_weight_emb, pos_ids,
                    mask, layer_past, use_cache, teacher_state
                )
                
            aux_losses.append(layer_aux)
            student_hidden_states.append(x)
            if use_cache and current_kv is not None:
                new_key_values.append(
                    tuple(t.contiguous() for t in current_kv)
                )

        total_aux_loss = torch.stack(aux_losses).sum()
        x      = self.norm_final(x)
        logits = self.lm_head(x.float() if x.dtype != torch.float32 else x)

        return {
            "logits":          logits,
            "hidden_states":   student_hidden_states,
            "aux_loss":        total_aux_loss,
            "past_key_values": new_key_values,
        }

    def project_state(
        self, layer_idx: int, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Project a student hidden state to teacher dimensionality."""
        return self.align_projs[layer_idx](hidden_state)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.875,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Autoregressively generate tokens using the KV cache with repetition penalty."""
        self.eval()
        device   = input_ids.device
        curr_ids = input_ids.clone()

        # Prefill
        outputs           = self.forward(curr_ids, use_cache=True)
        past_key_values   = outputs["past_key_values"]
        next_token_logits = outputs["logits"][:, -1, :]

        if device.type == "cuda":
            torch.cuda.empty_cache()

        next_token = self._sample(
            next_token_logits, do_sample, temperature, curr_ids, repetition_penalty
        )
        curr_ids   = torch.cat([curr_ids, next_token], dim=1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            return curr_ids

        # Decoding loop
        for _ in range(max_new_tokens - 1):
            outputs           = self.forward(
                next_token, past_key_values=past_key_values, use_cache=True
            )
            past_key_values   = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]
            next_token        = self._sample(
                next_token_logits, do_sample, temperature, curr_ids, repetition_penalty
            )
            curr_ids          = torch.cat([curr_ids, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return curr_ids

    def _sample(
        self,
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        curr_ids: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Select the next token via sampling or greedy decoding."""
        if repetition_penalty != 1.0 and curr_ids is not None:
            # Apply dynamic inference repetition penalty
            for i in range(logits.shape[0]):
                unique_tokens = torch.unique(curr_ids[i])
                score = logits[i, unique_tokens]
                
                # Make negative logits more negative, and positive logits less positive
                score = torch.where(
                    score < 0, score * repetition_penalty, score / repetition_penalty
                )
                logits[i, unique_tokens] = score

        if do_sample and temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)


# ==============================================================================
# 5. DISCRIMINATOR, DAGGER & ILP AGENT
# ==============================================================================
class SequenceDiscriminator(nn.Module):
    """GRU-based discriminator to distinguish teacher from student sequences.

    Used in a GAIL-style framework to provide additional training signal.

    Args:
        None — dimensions read from Config.
    """

    def __init__(self) -> None:
        super().__init__()
        vocab_size = Config.get_vocab_size()
        self.embed = nn.Embedding(vocab_size, Config.DISCRIMINATOR_HIDDEN_DIM)
        self.gru   = nn.GRU(
            Config.DISCRIMINATOR_HIDDEN_DIM,
            Config.DISCRIMINATOR_HIDDEN_DIM,
            batch_first=True,
        )
        self.head  = nn.Linear(Config.DISCRIMINATOR_HIDDEN_DIM, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Classify input_ids as teacher-like (high) or student-like (low).

        Args:
            input_ids: Integer token tensor (B, S).

        Returns:
            Logit tensor (B, 1).
        """
        x     = self.embed(input_ids)
        _, h  = self.gru(x)
        return self.head(h[-1])


class DaggerMixer:
    """Annealed dataset aggregation (DAgger) policy mixer.

    With probability beta the teacher's action is used; otherwise the student's.
    Beta decays towards a minimum over training steps.
    """

    def __init__(self) -> None:
        self.beta = Config.DAGGER_BETA_START
        self.step = 0

    def update(self) -> None:
        """Decay beta by DAGGER_BETA_DECAY, floored at DAGGER_BETA_MIN."""
        self.step += 1
        self.beta = max(
            Config.DAGGER_BETA_MIN,
            Config.DAGGER_BETA_START * (Config.DAGGER_BETA_DECAY ** self.step),
        )

    def mix_action(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the next token from a mixture of teacher and student policies.

        Args:
            student_logits: Student logits (B, vocab_size).
            teacher_logits: Teacher logits (B, vocab_size).

        Returns:
            Selected next-token IDs (B, 1).
        """
        student_token = torch.multinomial(F.softmax(student_logits, dim=-1), 1)
        teacher_token = torch.multinomial(F.softmax(teacher_logits, dim=-1), 1)
        use_teacher   = torch.bernoulli(
            torch.full_like(student_token, self.beta, dtype=torch.float)
        ).bool()
        return torch.where(use_teacher, teacher_token, student_token)


class ILPMCTSNode:
    """Monte-Carlo Tree Search Node for discovering optimal logical sub-sequences.

    Navigates the combinatorial space of trajectory indices to extract the
    highest-reward rule sequences for the ILP meta-planner.
    """

    def __init__(
        self,
        indices: List[int],
        target_len: int,
        max_idx: int,
        parent: Optional['ILPMCTSNode'] = None
    ) -> None:
        """Initialize the MCTS node.

        Args:
            indices:    List of trajectory indices selected so far.
            target_len: Target length of the program (sub-sequence).
            max_idx:    Maximum index available in the source trajectory.
            parent:     Parent node (None for root).
        """
        self.indices    = indices
        self.target_len = target_len
        self.max_idx    = max_idx
        self.parent     = parent
        self.children: List['ILPMCTSNode'] = []
        self.visits: int   = 0
        self.value: float  = 0.0

        last_idx         = indices[-1] if indices else -1
        needed_remaining = target_len - len(indices) - 1

        if len(indices) >= target_len:
            self.untried_actions = []
        else:
            self.untried_actions = list(range(
                last_idx + 1,
                max_idx - needed_remaining
            ))

    def is_fully_expanded(self) -> bool:
        """Check if all possible child actions have been explored."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if the node has reached the target program length."""
        return len(self.indices) == self.target_len

    def expand(self) -> 'ILPMCTSNode':
        """Randomly expand one of the untried actions."""
        if not self.untried_actions:
            raise ValueError("Cannot expand a fully expanded or terminal node.")

        action_idx  = random.randrange(len(self.untried_actions))
        action      = self.untried_actions.pop(action_idx)
        new_indices = self.indices + [action]

        child = ILPMCTSNode(
            indices=new_indices,
            target_len=self.target_len,
            max_idx=self.max_idx,
            parent=self
        )
        self.children.append(child)
        return child

    def best_child(self, c_param: float = 1.414) -> 'ILPMCTSNode':
        """Select the best child using the Upper Confidence Bound (UCB1) formula."""
        best_score      = -float('inf')
        best_candidates: List['ILPMCTSNode'] = []

        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.value / (child.visits + 1e-8)
                explore = c_param * math.sqrt(
                    math.log(self.visits + 1) / (child.visits + 1e-8)
                )
                score   = exploit + explore

            if score > best_score:
                best_score      = score
                best_candidates = [child]
            elif score == best_score:
                best_candidates.append(child)

        if not best_candidates:
            raise ValueError("Node has no children to select from.")

        return random.choice(best_candidates)


class PointerHead(nn.Module):
    """Pointer Network Head for selecting start/end indices from a sequence.

    Resolves the infinite action space issue by restricting the policy output
    to valid indices within the variable-length input sequence.

    Args:
        hidden_dim:  Dimension of the encoder output and internal projections.
        temperature: Softmax temperature for pointer attention.
    """

    def __init__(self, hidden_dim: int, temperature: float = 0.875) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.temperature = temperature

        self.query_start    = nn.Parameter(torch.randn(1, hidden_dim))
        self.query_end      = nn.Parameter(torch.randn(1, hidden_dim))
        self.key_proj       = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.start_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for start and end indices.

        Args:
            encoder_outputs: Tensor of shape (Batch, Seq_Len, Hidden_Dim).
            mask: Optional boolean mask (Batch, Seq_Len), True for valid tokens.

        Returns:
            Tuple of (start_logits, end_logits), each shape (Batch, Seq_Len).
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        keys = self.key_proj(encoder_outputs)

        q_start_expanded = self.query_start.unsqueeze(0).expand(batch_size, 1, -1)
        start_logits = torch.bmm(q_start_expanded, keys.transpose(1, 2)).squeeze(1)
        start_logits = start_logits / self.temperature

        if mask is not None:
            start_logits = start_logits.masked_fill(~mask, -1e9)

        start_probs  = F.softmax(start_logits, dim=-1)
        start_context = torch.bmm(start_probs.unsqueeze(1), encoder_outputs).squeeze(1)

        q_end_dynamic = (
            self.query_end.unsqueeze(0).expand(batch_size, -1, -1)
            + self.start_embedding(start_context).unsqueeze(1)
        )
        end_logits = torch.bmm(q_end_dynamic, keys.transpose(1, 2)).squeeze(1)
        end_logits = end_logits / self.temperature

        if mask is not None:
            end_logits = end_logits.masked_fill(~mask, -1e9)

        return start_logits, end_logits


class RLILPAgent:
    """Reinforcement-learning agent with ILP-based meta-planning and Pointer Networks.

    Combines a Transformer-based Actor-Critic with an Inductive Logic Programming
    meta-planner.  The Actor uses a Pointer Network to select sub-sequences (rules)
    from high-reward trajectories.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_program_size: int = 10,
        device: str = "cpu",
        use_mcts: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_program_size = max_program_size
        self.use_mcts = use_mcts

        self.input_proj = nn.Linear(obs_dim, hidden_dim).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.shared_brain = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        ).to(self.device)

        self.actor = PointerHead(hidden_dim).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.input_proj.apply(self._init_weights_module)
        self.shared_brain.apply(self._init_weights_module)
        self.actor.apply(self._init_weights_module)
        self.critic.apply(self._init_weights_module)

        self.optimizer = optim.AdamW(
            list(self.input_proj.parameters())
            + list(self.shared_brain.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5,
        )

        self.total_episodes: int = 0
        self.global_knowledge_base: Set[str] = set()
        self.ilp_positive_examples: List[torch.Tensor] = []
        self.ilp_negative_examples: List[torch.Tensor] = []
        self.current_logical_plan: Optional[List[torch.Tensor]] = None

    @staticmethod
    def _init_weights_module(module: nn.Module) -> None:
        """Orthogonal weight initialisation for Linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Atomically save agent state to a checkpoint file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        checkpoint = {
            "input_proj_state_dict":  self.input_proj.state_dict(),
            "shared_brain_state_dict": self.shared_brain.state_dict(),
            "actor_state_dict":       self.actor.state_dict(),
            "critic_state_dict":      self.critic.state_dict(),
            "optimizer_state_dict":   self.optimizer.state_dict(),
            "total_episodes":         self.total_episodes,
            "global_knowledge_base":  list(self.global_knowledge_base),
        }
        try:
            torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
            if path.exists():
                path.unlink()
            tmp_path.rename(path)
            logging.getLogger("RL-ILP-Agent").info("Checkpoint saved to %s", path)
        except (RuntimeError, OSError) as save_exc:
            logging.getLogger("RL-ILP-Agent").error(
                "Failed to save agent checkpoint: %s", save_exc
            )

    def load_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Load agent state from a checkpoint file."""
        path = Path(filepath)
        if not path.exists():
            logging.getLogger("RL-ILP-Agent").warning("Checkpoint %s not found.", path)
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.input_proj.load_state_dict(checkpoint["input_proj_state_dict"])
        self.shared_brain.load_state_dict(checkpoint["shared_brain_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_episodes = checkpoint["total_episodes"]
        self.global_knowledge_base = set(checkpoint["global_knowledge_base"])

    def get_action(
        self, trajectory_features: torch.Tensor
    ) -> Tuple[int, int, float, float]:
        """Select start and end indices for a rule using the Pointer Network.

        Args:
            trajectory_features: Tensor of shape (Seq_Len, Obs_Dim).

        Returns:
            Tuple of (start_idx, end_idx, start_log_prob, end_log_prob).
        """
        self.input_proj.eval()
        self.shared_brain.eval()
        self.actor.eval()

        with torch.no_grad():
            x     = trajectory_features.unsqueeze(0).to(self.device)
            h     = self.input_proj(x)
            h_enc = self.shared_brain(h)

            start_logits, end_logits = self.actor(h_enc)
            seq_len = start_logits.shape[1]

            start_dist = torch.distributions.Categorical(logits=start_logits)
            start_idx  = start_dist.sample()

            positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
            mask_end  = positions >= start_idx.unsqueeze(1)
            end_logits = end_logits.masked_fill(~mask_end, -1e9)

            end_dist = torch.distributions.Categorical(logits=end_logits)
            end_idx  = end_dist.sample()

            return (
                start_idx.item(),
                end_idx.item(),
                start_dist.log_prob(start_idx).item(),
                end_dist.log_prob(end_idx).item(),
            )

    def _process_episode(
        self,
        trajectory_features: List[np.ndarray],
        trajectory_actions: List[Tuple[int, int]],
        trajectory_rewards: List[float],
        trajectory_dones: List[float],
        trajectory_values: List[float],
        trajectory_log_probs: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        """Run one PPO update step on completed episode trajectories."""
        self.total_episodes += 1

        mean_reward = np.mean(trajectory_rewards)
        for i, features_np in enumerate(trajectory_features):
            t_features = torch.tensor(features_np, dtype=torch.float32, device="cpu")
            if trajectory_rewards[i] > ILP_COVERAGE_THRESHOLD:
                self.ilp_positive_examples.append(t_features)
            else:
                self.ilp_negative_examples.append(t_features)

        if self.ilp_positive_examples:
            self._run_ilp_meta_planner()

        self.input_proj.train()
        self.shared_brain.train()
        self.actor.train()
        self.critic.train()

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        batch_count       = len(trajectory_features)

        self.optimizer.zero_grad()

        for i in range(batch_count):
            feat = torch.tensor(
                trajectory_features[i], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            start_act, end_act       = trajectory_actions[i]
            old_lp_start, old_lp_end = trajectory_log_probs[i]
            reward                   = trajectory_rewards[i]

            h      = self.input_proj(feat)
            h_enc  = self.shared_brain(h)
            value_pred = self.critic(h_enc.mean(dim=1)).squeeze(-1)

            s_logits, e_logits = self.actor(h_enc)
            seq_len = feat.shape[1]

            positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
            mask_end  = positions >= start_act
            e_logits  = e_logits.masked_fill(~mask_end, -1e9)

            dist_s = torch.distributions.Categorical(logits=s_logits)
            dist_e = torch.distributions.Categorical(logits=e_logits)

            t_start  = torch.tensor([start_act], device=self.device)
            t_end    = torch.tensor([end_act],   device=self.device)
            new_lp_s = dist_s.log_prob(t_start)
            new_lp_e = dist_e.log_prob(t_end)

            entropy   = (dist_s.entropy() + dist_e.entropy()).mean()
            advantage = reward - value_pred.detach()
            ratio_s   = torch.exp(new_lp_s - old_lp_start)
            ratio_e   = torch.exp(new_lp_e - old_lp_end)
            ratio     = (ratio_s + ratio_e) / 2.0

            surr1       = ratio * advantage
            surr2       = torch.clamp(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            ) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            target_val  = torch.tensor([reward], dtype=torch.float32, device=self.device)
            value_loss  = F.mse_loss(value_pred, target_val)
            loss        = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )
            (loss / batch_count).backward()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()

        torch.nn.utils.clip_grad_norm_(
            list(self.shared_brain.parameters()) + list(self.actor.parameters()), 0.5
        )
        self.optimizer.step()

        if self.total_episodes % 50 == 0 and self.total_episodes > 0:
            self.save_checkpoint(f"checkpoints/agent_ep_{self.total_episodes}.pth")

        return {
            "total_loss":   (total_policy_loss + total_value_loss) / batch_count,
            "policy_loss":  total_policy_loss / batch_count,
            "value_loss":   total_value_loss / batch_count,
            "mean_reward":  mean_reward,
            "entropy":      total_entropy / batch_count,
        }

    def _violates_ilp_constraints(self, state_tensor: torch.Tensor) -> bool:
        """Check if state_tensor is too far from any known positive example."""
        if self.ilp_positive_examples:
            known_valid = torch.cat(self.ilp_positive_examples)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            min_dist = torch.cdist(state_tensor.cpu(), known_valid).min()
            if min_dist > 5.0:
                return True
        return False

    def _get_program_hash(self, program: List[torch.Tensor]) -> str:
        """Create a hash string for a list of tensors."""
        return str(hash(sum(t.sum().item() for t in program)))

    def _run_ilp_meta_planner(self) -> None:
        """Run the ILP meta-planner to extract symbolic rules from examples."""
        if len(self.ilp_positive_examples) < 2:
            return

        candidates: List[Dict[str, Any]] = []
        score_best = float("inf")
        h_best     = None

        max_len_in_examples = max(t.shape[0] for t in self.ilp_positive_examples)
        max_k = min(self.max_program_size, max_len_in_examples)

        for k in range(1, max_k + 1):
            local_constraints: Set[str] = set()

            if self.use_mcts:
                num_templates = min(3, len(self.ilp_positive_examples))
                sampled       = random.sample(self.ilp_positive_examples, num_templates)

                for template in sampled:
                    candidate = self._mcts_generate_program(
                        template=template,
                        size=k,
                        pos_examples=self.ilp_positive_examples,
                        neg_examples=self.ilp_negative_examples,
                        local_const=local_constraints,
                        num_simulations=40,
                    )
                    if candidate is None:
                        continue

                    tp, fp, fn, inconsistent = self._test_program(
                        candidate,
                        self.ilp_positive_examples,
                        self.ilp_negative_examples,
                    )
                    if tp == len(self.ilp_positive_examples) and fp == 0 and not inconsistent:
                        h_best     = candidate
                        score_best = 0.0
                        break

                    prog_hash = self._get_program_hash(candidate)
                    if tp == 0:
                        local_constraints.add(f"Prune_Spec_{prog_hash}")
                    elif inconsistent:
                        local_constraints.add(f"Prune_Gen_{prog_hash}")
                    if not inconsistent and tp > 0:
                        candidates.append(
                            {"program": candidate, "tp": tp, "fp": fp, "fn": fn, "size": k}
                        )
            else:
                for _ in range(5 * k):
                    candidate = self._generate_simulated_program(
                        k, set(self.global_knowledge_base), local_constraints
                    )
                    if candidate is None:
                        continue

                    tp, fp, fn, inconsistent = self._test_program(
                        candidate,
                        self.ilp_positive_examples,
                        self.ilp_negative_examples,
                    )
                    if tp == len(self.ilp_positive_examples) and fp == 0 and not inconsistent:
                        h_best     = candidate
                        score_best = 0.0
                        break

                    prog_hash = self._get_program_hash(candidate)
                    if tp == 0:
                        local_constraints.add(f"Prune_Spec_{prog_hash}")
                    elif inconsistent or tp == len(self.ilp_positive_examples):
                        local_constraints.add(f"Prune_Gen_{prog_hash}")
                    if not inconsistent and tp > 0:
                        candidates.append(
                            {"program": candidate, "tp": tp, "fp": fp, "fn": fn, "size": k}
                        )

            if score_best == 0.0:
                break

        if candidates and h_best is None:
            best_cand_dict = self._simulate_combiner_maxsat(candidates)
            if best_cand_dict["score"] < score_best:
                h_best = best_cand_dict["program"]

        if h_best is not None:
            self.current_logical_plan = h_best
            self._extract_and_broadcast_metarules(h_best)
            self.ilp_positive_examples.clear()
            self.ilp_negative_examples.clear()

    def _mcts_generate_program(
        self,
        template: torch.Tensor,
        size: int,
        pos_examples: List[torch.Tensor],
        neg_examples: List[torch.Tensor],
        local_const: Set[str],
        num_simulations: int = 50,
    ) -> Optional[List[torch.Tensor]]:
        """Use MCTS to find the optimal sub-sequence of length `size`."""
        T = template.shape[0]
        if T < size:
            return None

        root = ILPMCTSNode(indices=[], target_len=size, max_idx=T)

        for _ in range(num_simulations):
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child()
            if not node.is_terminal():
                node = node.expand()

            rollout_indices = node.indices.copy()
            while len(rollout_indices) < size:
                last_idx      = rollout_indices[-1] if rollout_indices else -1
                needed        = size - len(rollout_indices) - 1
                possible_next = list(range(last_idx + 1, T - needed))
                if not possible_next:
                    break
                rollout_indices.append(random.choice(possible_next))

            if len(rollout_indices) < size:
                continue

            candidate_program = [template[i] for i in rollout_indices]
            prog_hash         = self._get_program_hash(candidate_program)
            is_pruned         = (
                any(self._violates_ilp_constraints(s.unsqueeze(0)) for s in candidate_program)
                or f"Prune_Spec_{prog_hash}" in local_const
                or f"Prune_Gen_{prog_hash}" in local_const
            )

            if is_pruned:
                reward = -1.0
            else:
                tp, fp, fn, inconsistent = self._test_program(
                    candidate_program, pos_examples, neg_examples
                )
                tp_rate = tp / max(len(pos_examples), 1)
                fp_rate = fp / max(len(neg_examples), 1)
                reward  = tp_rate - (2.0 * fp_rate)
                if inconsistent:
                    reward -= 0.5

            while node is not None:
                node.visits += 1
                node.value  += reward
                node         = node.parent

        best_node = root
        while not best_node.is_terminal():
            if not best_node.children:
                return None
            best_node = best_node.best_child(c_param=0.0)

        return [template[i] for i in best_node.indices]

    def _generate_simulated_program(
        self,
        size: int,
        global_const: Set[str],
        local_const: Set[str],
    ) -> Optional[List[torch.Tensor]]:
        """Generate a candidate ILP program by sub-sampling a positive example."""
        with torch.no_grad():
            try:
                valid = [t for t in self.ilp_positive_examples if t.shape[0] >= size]
                if not valid:
                    return None
                template = random.choice(valid)
                indices  = sorted(random.sample(range(template.shape[0]), size))
                program  = [template[i] for i in indices]
                for state in program:
                    if self._violates_ilp_constraints(state.unsqueeze(0)):
                        return None
                prog_hash = self._get_program_hash(program)
                if (
                    f"Prune_Spec_{prog_hash}" in local_const
                    or f"Prune_Gen_{prog_hash}" in local_const
                ):
                    return None
                return program
            except (IndexError, RuntimeError, ValueError):
                return None

    def _test_program(
        self,
        program: List[torch.Tensor],
        pos_examples: List[torch.Tensor],
        neg_examples: List[torch.Tensor],
    ) -> Tuple[int, int, int, bool]:
        """Evaluate a program against positive and negative examples."""
        tp = sum(1 for ex in pos_examples if self._is_covered(program, ex))
        fp = sum(1 for ex in neg_examples if self._is_covered(program, ex))
        return tp, fp, len(pos_examples) - tp, fp > 0

    def _is_covered(
        self, program: List[torch.Tensor], trajectory: torch.Tensor
    ) -> bool:
        """Check whether a trajectory covers all program clauses in order."""
        prog_idx = traj_idx = 0
        while prog_idx < len(program) and traj_idx < trajectory.shape[0]:
            if torch.norm(trajectory[traj_idx] - program[prog_idx]) < 1.0:
                prog_idx += 1
            traj_idx += 1
        return prog_idx == len(program)

    def _simulate_combiner_maxsat(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select the best candidate program by weighted MaxSAT heuristic."""
        best       = candidates[0] if candidates else {}
        best_score = float("inf")
        for cand in candidates:
            score = cand["fp"] + cand["fn"] + (0.1 * cand["size"])
            cand["score"] = score
            if score < best_score:
                best_score = score
                best       = cand
        return best

    def _extract_and_broadcast_metarules(self, h_best: List[torch.Tensor]) -> None:
        """Register the extracted program as a named metarule in the knowledge base."""
        prog_hash = self._get_program_hash(h_best)
        rule_sig  = f"Rule_{prog_hash}"
        if rule_sig not in self.global_knowledge_base:
            self.global_knowledge_base.add(rule_sig)
            logging.getLogger("RL-ILP-Agent").info(
                "Extracted new Metarule: %s", rule_sig
            )


class SemanticTextEncoder(nn.Module):
    """Sentence-level semantic encoder using a frozen MiniLM model.

    Encodes a batch of strings into normalised embedding vectors for use
    in the ILP meta-planner.

    Args:
        model_name: HuggingFace model identifier.
        device:     Compute device string.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        logging.getLogger("RL-ILP-Agent").info(
            "Loading SemanticTextEncoder: %s ...", model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sentences: List[str]) -> torch.Tensor:
        """Encode a batch of sentences into mean-pooled, L2-normalised embeddings.

        Args:
            sentences: List of input strings.

        Returns:
            Embedding tensor (len(sentences), hidden_dim).
        """
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output[0]
        attention_mask   = encoded_input["attention_mask"]
        mask_expanded    = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings   = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask         = mask_expanded.sum(1).clamp(min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        return F.normalize(sentence_embeddings, p=2, dim=1)


# ==============================================================================
# 6. TRAINING SYLLABUS
# ==============================================================================
# Prompts are plain content strings. train_step() calls tokenizer.apply_chat_template()
# exactly once, so chat-template markers must NOT appear in the prompt text itself.
# ==============================================================================
STEPS_PER_TIER: int     = 300
PROMPTS_PER_TIER: int   = 4
NUM_TIERS: int          = 4
SYLLABUS_CYCLE_LEN: int = STEPS_PER_TIER * NUM_TIERS

_EASY_PROMPTS: List[str] = [
    "Write a Python function that takes a list of integers and returns their sum."
    " Include a docstring.",
    "Write a Python function that reverses a string without using slicing.",
    "Write a Python function that checks whether a given number is even or odd"
    " and returns a boolean.",
    "Write a Python function that converts a temperature from Celsius to Fahrenheit.",
]

_MID_PROMPTS: List[str] = [
    "Implement a Python class `Stack` using a list as the backing store."
    " Include push, pop, peek, and is_empty methods with proper type hints.",
    "Write a Python generator function that yields the Fibonacci sequence up to"
    " a given limit n. Use memoisation via functools.lru_cache.",
    "Write a Python context manager class `Timer` that records the elapsed"
    " wall-clock time of any code block placed inside a `with` statement.",
    "Write a Python function that reads a CSV file with pathlib.Path, parses it"
    " with the csv module, and returns a list of dicts mapping header names to row values.",
]

_HARD_PROMPTS: List[str] = [
    "Implement a thread-safe LRU cache in Python using collections.OrderedDict"
    " and threading.Lock. The class must support get(key), put(key, value), and"
    " a configurable capacity. Include full type hints and docstrings.",
    "Write an async Python web scraper using aiohttp and asyncio.gather that"
    " fetches multiple URLs concurrently, handles HTTP errors, and returns a dict"
    " mapping each URL to its response text.",
    "Implement Dijkstra's shortest-path algorithm in Python using a min-heap"
    " (heapq). The graph is represented as an adjacency list of dicts. Return both"
    " the shortest distances and the predecessor map for path reconstruction.",
    "Write a Python metaclass `SingletonMeta` that enforces the singleton pattern"
    " for any class that uses it as its metaclass. It must be thread-safe and"
    " support subclassing.",
]

_PRO_PROMPTS: List[str] = [
    "Design and implement a Python library for a distributed task queue. Include a"
    " `TaskBroker` class that uses Redis pub/sub for task dispatch, a `Worker` class"
    " that processes tasks concurrently with asyncio, dead-letter queue handling for"
    " failed tasks, and an exponential backoff retry policy. Provide full type"
    " annotations and docstrings.",
    "Implement a lock-free, wait-free concurrent hash map in Python using"
    " multiprocessing.shared_memory and atomic CAS operations. Support get, put,"
    " delete, and resize. Analyse the amortised time complexity of each operation.",
    "Write a Python JIT compiler micro-kernel that takes a simple arithmetic"
    " expression DSL (e.g. '3 * x + sin(x)'), parses it into an AST, applies"
    " constant folding and dead-code elimination, and emits Python bytecode via the"
    " `bytecode` library. Include a benchmark harness comparing interpreted vs"
    " compiled execution.",
    "Implement a differentiable neural ODE solver in pure PyTorch (no torchdiffeq)."
    " Use the adjoint sensitivity method for gradient computation, support both"
    " fixed-step Euler and adaptive RK4(5) integrators, and demonstrate training on"
    " a simple 2-D spiral classification task.",
]

_TIER_REGISTRY: List[Tuple[str, List[str]]] = [
    ("EASY", _EASY_PROMPTS),
    ("MID",  _MID_PROMPTS),
    ("HARD", _HARD_PROMPTS),
    ("PRO",  _PRO_PROMPTS),
]

for _tier_label, _tier_prompts in _TIER_REGISTRY:
    assert len(_tier_prompts) == PROMPTS_PER_TIER, (
        f"Tier '{_tier_label}' must have exactly {PROMPTS_PER_TIER} prompts; "
        f"found {len(_tier_prompts)}."
    )


def _resolve_tier_and_position(global_step: int) -> Tuple[int, int]:
    """Map a global training step to a tier index and position within that tier.

    Args:
        global_step: Non-negative integer training step.

    Returns:
        Tuple of (tier_index, position_within_tier).

    Raises:
        ValueError: If global_step is negative.

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    if global_step < 0:
        raise ValueError(
            f"global_step must be non-negative; received {global_step}."
        )
    step_in_cycle: int = global_step % SYLLABUS_CYCLE_LEN
    tier_index: int    = step_in_cycle // STEPS_PER_TIER
    position: int      = step_in_cycle  % STEPS_PER_TIER
    return tier_index, position


def get_prompt(global_step: int) -> str:
    """Return the plain-content training prompt for a given global step.

    The returned string contains only the user-visible question text.
    Callers are responsible for wrapping it in a chat-template message dict
    (e.g. ``[{"role": "user", "content": get_prompt(step)}]``) before
    passing to ``tokenizer.apply_chat_template()``.

    Args:
        global_step: Non-negative integer training step.

    Returns:
        Plain prompt string for the step's tier and position.
    """
    tier_index, position = _resolve_tier_and_position(global_step)
    tier_label, tier_prompts = _TIER_REGISTRY[tier_index]
    prompt_index: int = position % PROMPTS_PER_TIER
    prompt: str       = tier_prompts[prompt_index]
    logging.getLogger("RL-ILP-Agent").debug(
        "Step %d → Tier %d (%s), position %d, prompt_index %d",
        global_step, tier_index, tier_label, position, prompt_index,
    )
    return prompt


def get_tier_name(global_step: int) -> str:
    """Return the tier label for a given global step.

    Args:
        global_step: Non-negative integer training step.

    Returns:
        Tier label string (e.g. "EASY", "MID", "HARD", "PRO").
    """
    tier_index, _ = _resolve_tier_and_position(global_step)
    return _TIER_REGISTRY[tier_index][0]


def get_syllabus_summary() -> str:
    """Build a human-readable summary of the training syllabus.

    Returns:
        Multi-line string describing all tiers and their prompts.
    """
    lines: List[str] = [
        "=" * 72,
        "DEEPMIME TRAINING SYLLABUS",
        f"Cycle length : {SYLLABUS_CYCLE_LEN} steps  "
        f"({NUM_TIERS} tiers × {STEPS_PER_TIER} steps each)",
        f"Prompts/tier : {PROMPTS_PER_TIER}  "
        f"(each prompt repeats {STEPS_PER_TIER // PROMPTS_PER_TIER}× per window)",
        "=" * 72,
    ]
    for tier_idx, (tier_label, tier_prompts) in enumerate(_TIER_REGISTRY):
        step_start = tier_idx * STEPS_PER_TIER
        step_end   = step_start + STEPS_PER_TIER - 1
        lines.append(
            f"Tier {tier_idx}  [{tier_label:4s}]  "
            f"steps {step_start:>4d} – {step_end:>4d}  (within each cycle)"
        )
        lines.append("-" * 72)
        for p_idx, prompt_text in enumerate(tier_prompts):
            display = prompt_text.replace("\n", " ")
            if len(display) > 90:
                display = display[:87] + "..."
            lines.append(f"  [{p_idx}] {display}")
        lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ==============================================================================
# 7. UTILS & COMPONENTS
# ==============================================================================
try:
    from timm.utils import ModelEmaV2 as ModelEma
except ImportError:
    class ModelEma:  # type: ignore[no-redef]
        """Minimal EMA fallback when timm is not installed."""

        def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
            self.module = model
            self.decay  = decay

        # Proper method signature for EMA update
        def update(self, model: nn.Module) -> None:
            """No-op EMA update (fallback; real EMA requires timm).

            Args:
                model: Source model whose weights would be averaged in.
            """


class ExperienceReplayBuffer:
    """Fixed-capacity FIFO buffer for storing high-reward trajectory entries.

    Args:
        capacity: Maximum number of entries to retain.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity  = capacity
        self._entries: List[Dict[str, Any]] = []

    def insert(
        self,
        trajectory_ids:   torch.Tensor,
        prompt_length:    int,
        reward:           float,
        source_log_probs: torch.Tensor,
        timestamp:        int,
    ) -> None:
        """Insert a new trajectory entry, evicting the oldest if over capacity.

        Args:
            trajectory_ids:   Full sequence token IDs (S,).
            prompt_length:    Number of prompt tokens (generation starts here).
            reward:           Scalar reward for this trajectory.
            source_log_probs: Per-token log-probabilities at insertion time (S,).
            timestamp:        Training step at which this entry was produced.
        """
        entry = {
            "trajectory_ids":   trajectory_ids.detach().cpu(),
            "prompt_length":    prompt_length,
            "reward":           reward,
            "source_log_probs": source_log_probs.detach().cpu(),
            "timestamp":        timestamp,
        }
        self._entries.append(entry)
        while len(self._entries) > self.capacity:
            self._entries.pop(0)

    def sample(self, count: int) -> List[Dict[str, Any]]:
        """Sample up to count entries without replacement.

        Args:
            count: Number of entries to sample.

        Returns:
            List of sampled entry dicts (may be empty if buffer is empty).
        """
        if not self._entries:
            return []
        actual_count = min(count, len(self._entries))
        indices      = np.random.choice(len(self._entries), actual_count, replace=False)
        return [self._entries[i] for i in indices]

    def __len__(self) -> int:
        """Return the current number of entries in the buffer."""
        return len(self._entries)

    def state_dict(self) -> Dict[str, Any]:
        """Serialise buffer state for checkpointing."""
        return {"entries": self._entries, "capacity": self.capacity}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore buffer state from a checkpoint dict."""
        self._entries = state.get("entries", [])
        self.capacity = state.get("capacity", self.capacity)


# ── Completion predicates (LUMOS-GRPO) ───────────────────────────────────────

def predicate_has_non_trivial_content(decoded_text: str) -> bool:
    """Return True if decoded_text has enough characters and unique symbols.

    Args:
        decoded_text: Decoded generation string.

    Returns:
        True when length ≥ _MINIMUM_CONTENT_CHARS and unique chars ≥ _MINIMUM_UNIQUE_CHARS.
    """
    stripped = decoded_text.strip()
    return (
        len(stripped) >= Config._MINIMUM_CONTENT_CHARS
        and len(set(stripped)) >= Config._MINIMUM_UNIQUE_CHARS
    )


def predicate_has_minimum_generation_length(generated_token_count: int) -> bool:
    """Return True if generated_token_count meets the minimum threshold.

    Args:
        generated_token_count: Number of generated (non-prompt) tokens.

    Returns:
        True when count ≥ _MINIMUM_GEN_TOKENS.
    """
    return generated_token_count >= Config._MINIMUM_GEN_TOKENS


def predicate_has_structural_completion(decoded_text: str) -> bool:
    """Return True if the generation ends with a sentence-closing character.

    Args:
        decoded_text: Decoded generation string.

    Returns:
        True when the last non-whitespace character is in _SENTENCE_ENDINGS.
    """
    stripped = decoded_text.rstrip()
    return bool(stripped) and stripped[-1] in Config._SENTENCE_ENDINGS


# ── Constraint functions & budgets ───────────────────────────────────────────

_CONSTRAINT_BUDGETS: Dict[str, float] = {
    "format_compliance":  0.05,
    "repetition_penalty": 0.10,
}


def constraint_format_compliance(decoded_text: str) -> int:
    """Return 1 (violated) if the text is empty or contains only control chars.

    Args:
        decoded_text: Decoded generation string.

    Returns:
        1 if format is violated, else 0.
    """
    stripped = decoded_text.strip()
    if not stripped:
        return 1
    if all(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        return 1
    return 0


def constraint_repetition_penalty(decoded_text: str) -> int:
    """Return 1 (violated) if the text has a unique-character ratio below 0.05.

    Args:
        decoded_text: Decoded generation string.

    Returns:
        1 if repetition threshold is exceeded, else 0.
    """
    if len(decoded_text) < 10:
        return 0
    unique_ratio = len(set(decoded_text)) / max(len(decoded_text), 1)
    return 1 if unique_ratio < 0.05 else 0


_CONSTRAINT_FUNCTIONS: Dict[str, Any] = {
    "format_compliance":  constraint_format_compliance,
    "repetition_penalty": constraint_repetition_penalty,
}


# ==============================================================================
# 7b. ENTROPY CONTROLLER & RL WARMUP
# ==============================================================================

class AdaptiveEntropyController:
    """Adaptive entropy-bonus coefficient controller for GRPO training.

    Increases the entropy-bonus coefficient (alpha) when entropy drops below
    the target, providing a corrective force that prevents irreversible collapse.

    Args:
        target_entropy_nats: Minimum healthy entropy (nats).
        alpha_init:          Starting bonus coefficient.
        delta:               Step size for coefficient updates.
        alpha_max:           Hard ceiling to prevent entropy-bonus domination.

    Time complexity per update: O(1).
    Space complexity: O(1).
    """

    def __init__(
        self,
        target_entropy_nats: float = ENTROPY_TARGET_NATS,
        alpha_init: float           = ENTROPY_ALPHA_INIT,
        delta: float                = ENTROPY_ALPHA_DELTA,
        alpha_max: float            = 0.5,
    ) -> None:
        self.target    = target_entropy_nats
        self.alpha     = alpha_init
        self.delta     = delta
        self.alpha_max = alpha_max

    def step(self, current_entropy_nats: float) -> float:
        """Update the coefficient based on current entropy and return it.

        Args:
            current_entropy_nats: Mean per-token entropy measured this step.

        Returns:
            Updated alpha coefficient to use as the entropy-bonus weight.
        """
        if current_entropy_nats < self.target:
            self.alpha = min(self.alpha + self.delta, self.alpha_max)
        else:
            self.alpha = max(0.0, self.alpha - self.delta)
        return self.alpha

    def state_dict(self) -> Dict[str, float]:
        """Return serialisable state for checkpointing."""
        return {
            "alpha":     self.alpha,
            "target":    self.target,
            "delta":     self.delta,
            "alpha_max": self.alpha_max,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        """Restore controller state from a checkpoint dict."""
        self.alpha     = float(state.get("alpha", self.alpha))
        self.target    = float(state.get("target", self.target))
        self.delta     = float(state.get("delta", self.delta))
        self.alpha_max = float(state.get("alpha_max", self.alpha_max))


def get_rl_grpo_weight(step: int) -> float:
    """Return the GRPO loss weight for the current training step.

    Applies a linear RL warmup. GRPO weight is 0.0 during the first
    RL_WARMUP_STEPS steps, then ramps linearly to RL_MAX_GRPO_WEIGHT.

    Args:
        step: Current global training step (1-indexed).

    Returns:
        GRPO loss weight in [0.0, RL_MAX_GRPO_WEIGHT].

    Time complexity:  O(1).
    Space complexity: O(1).
    """
    if step <= RL_WARMUP_STEPS:
        return 0.0
    ramp_steps = RL_WARMUP_STEPS
    if step <= RL_WARMUP_STEPS + ramp_steps:
        progress = (step - RL_WARMUP_STEPS) / ramp_steps
        return RL_MAX_GRPO_WEIGHT * progress
    return RL_MAX_GRPO_WEIGHT


# ==============================================================================
# 8. TRAINER
# ==============================================================================
class DeepMimeTrainer:
    """End-to-end trainer for DeepMime with LAA-GiGPO (DynaCogniual policy).

    Combines supervised distillation, hidden-state alignment, LAA-GiGPO nested-group
    policy optimisation (micro/macro advantages + Carousel Memory Alignment),
    experience replay, Lagrangian constraint enforcement, and an ILP meta-planner.

    Args:
        total_training_steps: Total number of train_step() calls planned.
        use_mcts:             Whether to use MCTS for ILP rule extraction.
    """

    # Gradient accumulation steps for fresher, lower-variance gradients
    _GRAD_ACCUM_STEPS: int  = GRAD_ACCUM_STEPS
    _EMA_UPDATE_FREQ: int   = GRAD_ACCUM_STEPS
    _ILP_MAX_EXAMPLES: int  = 8
    _AMP_DTYPE: torch.dtype = torch.bfloat16

    def __init__(self, total_training_steps: int = 1250, use_mcts: bool = True) -> None:
        self.device              = torch.device(Config.DEVICE)
        self.total_training_steps = total_training_steps
        self.use_mcts             = use_mcts

        self.teacher = DeepSeekTeacher().to(self.device)
        if self.teacher.tokenizer.pad_token is None:
            self.teacher.tokenizer.pad_token       = self.teacher.tokenizer.eos_token
            self.teacher.tokenizer.pad_token_id    = self.teacher.tokenizer.eos_token_id
            self.teacher.tokenizer.truncation_side = "left"

        self.semantic_encoder = SemanticTextEncoder(device=Config.DEVICE)

        weight_cache = WeightEmbeddingCache(
            teacher_model=self.teacher.transformer,
            weight_names=Config.WEIGHT_MATRICES_TO_ENCODE,
            num_teacher_layers=Config.get_teacher_num_layers(),
            strategy=Config.WEIGHT_ENCODER_STRATEGY,
            device=self.device,
        )
        self.student = DeepMimeStudent().to(self.device)
        self.student.set_weight_cache(weight_cache)
        self.ema = ModelEma(self.student, decay=EMA_DECAY)

        # ── LAA-GiGPO hyper-parameters ───────────────────────────────────
        self.G           = ROLLOUT_GROUP_SIZE
        self.epsilon_lo  = 0.2
        self.epsilon_hi  = 0.28
        self.beta        = BETA_GRPO       # KL penalty coefficient
        self.eps_thresh  = 1e-5
        self.t_max       = 256
        self.replay_alpha  = 0.25
        self.replay_w_max  = 5.0

        self.experience_buffer = ExperienceReplayBuffer(
            capacity=REPLAY_BUFFER_CAPACITY
        )
        self.lagrangian_lr = LAGRANGIAN_LR
        self.lagrangian_multipliers: Dict[str, float] = {
            name: 0.0 for name in _CONSTRAINT_FUNCTIONS
        }
        self.lagrangian_eos = 0.0

        # Adaptive entropy controller
        self.entropy_controller = AdaptiveEntropyController(
            target_entropy_nats=ENTROPY_TARGET_NATS,
            alpha_init=ENTROPY_ALPHA_INIT,
            delta=ENTROPY_ALPHA_DELTA,
        )

        # ── Optimiser & scheduler ─────────────────────────────────────────
        _fused_available = (
            self.device.type == "cuda" and torch.cuda.is_available()
        )
        self.opt_student = torch.optim.AdamW(
            self.student.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,   # weight decay coefficient
            fused=_fused_available,
        )
        _scheduler_steps = max(
            total_training_steps // self._GRAD_ACCUM_STEPS, 1
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_student,
            max_lr=Config.LEARNING_RATE,
            total_steps=_scheduler_steps,
            pct_start=0.125,
            anneal_strategy="cos",
            div_factor=2.5,
            final_div_factor=10.0,
        )

        # ── AMP ───────────────────────────────────────────────────────────
        _amp_enabled = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)
        self._amp_ctx = torch.amp.autocast(
            device_type=self.device.type,
            dtype=self._AMP_DTYPE if _amp_enabled else torch.float32,
            enabled=_amp_enabled,
        )

        # ── State ─────────────────────────────────────────────────────────
        self.total_steps  = 0
        self._accum_step  = 0
        self._micro_step  = 0
        self.global_knowledge_base: Set[str]           = set()
        self.ilp_positive_examples: List[torch.Tensor] = []
        self.ilp_negative_examples: List[torch.Tensor] = []
        self.current_logical_plan:  Optional[List[torch.Tensor]] = None
        self.max_program_size:      int                = 10
        self.checkpoint_dir = Path("checkpoints")
        
        # ── oss function ─────────────────────────────────────────────────────────        
        self.kl_symmetry_weight = 0.3      # Weight for reverse KL (0.0 = forward only)
        self.kl_anneal_steps = 1000        # Steps to fully anneal KL penalty
        self.entropy_coef = 0.01           # Entropy bonus coefficient
        self.carousel_grad_clip = 5.0      # Gradient clipping for carousel loss
        self.global_step = 0               # Track training progress       

    # ── Checkpoint I/O ────────────────────────────────────────────────────────
    def save_checkpoint(self, filename: str) -> None:
        """Atomically save full trainer state to a checkpoint file and rotate old ones.
        
        This method includes robust logic to parse step numbers from filenames and
        strictly delete previous checkpoints to prevent storage exhaustion.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.checkpoint_dir / filename
        tmp_path   = final_path.with_suffix(".tmp")

        ema_state = None
        if hasattr(self, "ema") and self.ema is not None:
            if hasattr(self.ema, "state_dict"):
                ema_state = self.ema.state_dict()
            elif hasattr(self.ema, "module") and hasattr(self.ema.module, "state_dict"):
                ema_state = self.ema.module.state_dict()

        checkpoint = {
            "student_state_dict":    self.student.state_dict(),
            "optimizer_state_dict":  self.opt_student.state_dict(),
            "scheduler_state_dict":  self.scheduler.state_dict(),
            "scaler_state_dict":     self.scaler.state_dict(),
            "ema_state_dict":        ema_state,
            "total_steps":           self.total_steps,
            "global_knowledge_base": list(self.global_knowledge_base),
            "ilp_positive_examples": self.ilp_positive_examples,
            "ilp_negative_examples": self.ilp_negative_examples,
            "experience_buffer":     self.experience_buffer.state_dict(),
            "lagrangian_multipliers": dict(self.lagrangian_multipliers),
            "lagrangian_eos":         self.lagrangian_eos,
            "entropy_controller":     self.entropy_controller.state_dict(),
        }
        try:
            torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
            if final_path.exists():
                try:
                    final_path.unlink()
                except OSError:
                    pass
            tmp_path.rename(final_path)
            logging.getLogger("RL-ILP-Agent").info(
                "Checkpoint saved → %s", final_path
            )

            # Robust Rotation Logic
            if filename.startswith("deepmime_step_"):
                import re  # Imported strictly for rotation parsing
                
                # Keep only the 2 most recent checkpoints (aggressive cleanup)
                MAX_CHECKPOINTS_TO_KEEP: int = 2
                
                try:
                    all_step_checkpoints = list(self.checkpoint_dir.glob("deepmime_step_*.pth"))
                    
                    # Sort files by the step number extracted from filename
                    # This is more reliable than file modification time (st_mtime)
                    def _extract_step_from_name(p: Path) -> int:
                        match = re.search(r"deepmime_step_(\d+)\.pth", p.name)
                        return int(match.group(1)) if match else 0
                    
                    all_step_checkpoints.sort(key=_extract_step_from_name)

                    if len(all_step_checkpoints) > MAX_CHECKPOINTS_TO_KEEP:
                        # Identify strict subset of files to delete
                        num_to_delete = len(all_step_checkpoints) - MAX_CHECKPOINTS_TO_KEEP
                        files_to_delete = all_step_checkpoints[:num_to_delete]
                        
                        for old_ckpt in files_to_delete:
                            if old_ckpt.exists():
                                old_ckpt.unlink()
                                logging.getLogger("RL-ILP-Agent").info(
                                    "Rotation: Deleted previous checkpoint %s to free space.", 
                                    old_ckpt.name
                                )
                except (OSError, ValueError) as rotation_error:
                    logging.getLogger("RL-ILP-Agent").warning(
                        "Failed to rotate checkpoints: %s", rotation_error
                    )

        except (RuntimeError, OSError) as save_exc:
            logging.getLogger("RL-ILP-Agent").error(
                "CRITICAL: Failed to save checkpoint to %s. Error: %s",
                final_path, save_exc,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_checkpoint(self, filename: str, load_ema: bool = False) -> None:
        """Load trainer state from a previously saved checkpoint."""
        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            logging.getLogger("RL-ILP-Agent").warning(
                "Checkpoint %s not found.", filepath
            )
            return
        logging.getLogger("RL-ILP-Agent").info(
            "Loading checkpoint from %s ...", filepath
        )
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=True
            )
        except (RuntimeError, KeyError, ValueError) as load_exc:
            logging.getLogger("RL-ILP-Agent").error(
                "Failed to load checkpoint: %s", load_exc
            )
            return

        target_sd = (
            checkpoint["ema_state_dict"]
            if (load_ema and checkpoint.get("ema_state_dict") is not None)
            else checkpoint["student_state_dict"]
        )
        self.student.load_state_dict(target_sd)
        self.opt_student.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if checkpoint.get("ema_state_dict") is not None:
            ema_loader = getattr(self.ema, "load_state_dict", None)
            if ema_loader is None and hasattr(self.ema, "module"):
                ema_loader = getattr(self.ema.module, "load_state_dict", None)
            if ema_loader is not None:
                ema_loader(checkpoint["ema_state_dict"])

        self.total_steps   = checkpoint["total_steps"]
        self._accum_step   = self.total_steps
        self.global_knowledge_base = set(checkpoint.get("global_knowledge_base", []))
        self.ilp_positive_examples = checkpoint.get("ilp_positive_examples", [])
        self.ilp_negative_examples = checkpoint.get("ilp_negative_examples", [])

        if "experience_buffer" in checkpoint:
            self.experience_buffer.load_state_dict(checkpoint["experience_buffer"])
        if "lagrangian_multipliers" in checkpoint:
            self.lagrangian_multipliers = checkpoint["lagrangian_multipliers"]
        if "lagrangian_eos" in checkpoint:
            self.lagrangian_eos = checkpoint["lagrangian_eos"]
        if "entropy_controller" in checkpoint:
            self.entropy_controller.load_state_dict(checkpoint["entropy_controller"])

        logging.getLogger("RL-ILP-Agent").info(
            "Checkpoint loaded successfully. Steps: %d  Replay: %d  λ_eos: %.4f",
            self.total_steps, len(self.experience_buffer), self.lagrangian_eos,
        )

    # ── Loss helpers ──────────────────────────────────────────────────────────

    def compute_soft_reward(
        self,
        teacher_logits: torch.Tensor,
        generated_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute per-sequence soft reward as mean teacher log-prob plus shaping.

        Applies the causal-LM shift: teacher_logits[:, t, :] predicts the token
        at position t+1.  We therefore align slice_logits[:, gen_start:-1, :] with
        slice_labels[:, gen_start+1:] so the reward reflects actual next-token
        log-probabilities over the generated region.
        """
        gen_start = max(prompt_len - 1, 0)
        # CAUSAL-LM SHIFT: logit[t] predicts label[t+1].
        # teacher_logits[:, gen_start:-1, :] predicts generated_ids[:, gen_start+1:]
        slice_logits = teacher_logits[:, gen_start:-1, :]
        slice_labels = generated_ids[:, gen_start + 1:]
        if slice_logits.shape[1] == 0:
            return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

        log_probs       = F.log_softmax(slice_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=slice_labels.unsqueeze(-1)
        ).squeeze(-1)
        base_reward = token_log_probs.mean(dim=-1)

        pad_id         = self.teacher.tokenizer.pad_token_id
        # Use the same shifted region for shaping statistics.
        gen_tokens_cpu = generated_ids[:, gen_start + 1:].cpu()
        shaped_rewards = base_reward.clone()

        for i in range(gen_tokens_cpu.shape[0]):
            seq     = gen_tokens_cpu[i]
            non_pad = [t.item() for t in seq if t.item() != pad_id]
            if not non_pad:
                shaped_rewards[i] = shaped_rewards[i] - WS_PENALTY_COEFF
                continue

            decoded_chars = self.teacher.tokenizer.decode(
                non_pad, skip_special_tokens=True
            )
            total_chars = max(len(decoded_chars), 1)
            ws_chars    = sum(1 for c in decoded_chars if c.isspace())
            ws_fraction = ws_chars / total_chars
            ws_excess   = max(ws_fraction - 0.5, 0.0)
            ws_pen      = -float(WS_PENALTY_COEFF) * ws_excess
            unique_ratio = len(set(non_pad)) / max(len(non_pad), 1)
            div_bonus    = float(DIVERSITY_BONUS_COEFF) * unique_ratio
            len_bonus    = float(LENGTH_BONUS_MAX) * min(len(non_pad) / 20.0, 1.0)
            shaped_rewards[i] = shaped_rewards[i] + ws_pen + div_bonus + len_bonus

        return shaped_rewards

    def compute_supervised_loss(
        self,
        student_logits: torch.Tensor,
        teacher_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute cross-entropy loss between student logits and teacher token IDs.

        Applies the standard causal-LM shift: logit at position t was produced from
        input token t and therefore predicts the token at position t+1.
        We start supervising from the first generated token (position prompt_len),
        which is predicted by the logit at position prompt_len-1.

            shift_logits[:, k, :] → predicts → shift_labels[:, k]
            k = 0 corresponds to sequence position gen_start
            k = 0 corresponds to label position gen_start + 1
        """
        gen_start = max(prompt_len - 1, 0)
        # CAUSAL-LM SHIFT: logit[t] → label[t+1].
        # logits[:, gen_start:-1, :] predicts teacher_ids[:, gen_start+1:]
        gen_logits = student_logits[:, gen_start:-1, :]
        gen_labels = teacher_ids[:, gen_start + 1:]
        if gen_logits.shape[1] == 0:
            return student_logits.new_zeros(())
        return F.cross_entropy(
            gen_logits.reshape(-1, gen_logits.shape[-1]),
            gen_labels.reshape(-1),
            reduction="mean",
        )

    def get_log_probs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for selected label tokens.

        Applies the standard causal-LM shift: logit at position t predicts the
        token at position t+1, so we align logits[:, :-1, :] with labels[:, 1:].
        Position 0 of the returned tensor is zero-padded so the output shape
        (B, S) matches the input labels tensor exactly.

        Args:
            logits: Raw logit tensor (B, S, vocab_size).
            labels: Target token-ID tensor (B, S).

        Returns:
            Per-token log-probability tensor (B, S); position 0 is always 0.0.

        Time complexity:  O(B * S * vocab_size) for log_softmax.
        Space complexity: O(B * S).
        """
        # CAUSAL-LM SHIFT: logit[t] predicts label[t+1].
        shift_logits = logits[:, :-1, :]       # (B, S-1, V)
        shift_labels = labels[:, 1:]            # (B, S-1)
        log_probs    = F.log_softmax(shift_logits, dim=-1)
        token_lp     = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)                           # (B, S-1)
        # Prepend a zero column so the output has shape (B, S).
        # Position 0 carries no information and is excluded by the GRPO
        # mask (which is only 1.0 for generated positions >= prompt_len).
        pad = torch.zeros(
            token_lp.shape[0], 1, device=token_lp.device, dtype=token_lp.dtype
        )
        return torch.cat([pad, token_lp], dim=1)

    def _compute_completion_scores(
        self,
        decoded_texts: List[str],
        generated_token_counts: List[int],
    ) -> torch.Tensor:
        """Score each generation on three LUMOS completion predicates."""
        NUM_PREDICATES = 3
        scores = torch.zeros(len(decoded_texts), device="cpu")
        for idx, text in enumerate(decoded_texts):
            satisfied = sum([
                predicate_has_non_trivial_content(text),
                predicate_has_minimum_generation_length(generated_token_counts[idx]),
                predicate_has_structural_completion(text),
            ])
            scores[idx] = satisfied / NUM_PREDICATES
        return scores.to(self.device)

    def _compute_constraint_costs(
        self, decoded_texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate all registered constraint functions on the decoded texts."""
        all_costs: Dict[str, torch.Tensor] = {}
        for constraint_name, constraint_fn in _CONSTRAINT_FUNCTIONS.items():
            cost_values = [float(constraint_fn(text)) for text in decoded_texts]
            all_costs[constraint_name] = torch.tensor(
                cost_values, dtype=torch.float32, device=self.device
            )
        return all_costs

    def _compute_eos_costs(
        self,
        completion_scores: torch.Tensor,
        rollout_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Assign an EOS cost for sequences that terminated before completing."""
        eos_token_id = self.teacher.tokenizer.eos_token_id
        eos_costs    = torch.zeros(rollout_ids.shape[0], device=self.device)
        for idx in range(rollout_ids.shape[0]):
            gen_seq     = rollout_ids[idx]
            eos_indices = (gen_seq == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                eos_costs[idx] = 1.0 - completion_scores[idx].item()
        return eos_costs

    def _compute_gigpo_advantages(
        self,
        rewards: torch.Tensor,
        constraint_costs: Dict[str, torch.Tensor],
        eos_costs: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Compute LAA-GiGPO nested-group advantages (micro + macro) with Lagrangian terms.

        The B rollouts are partitioned into K = GIGPO_LATENT_GROUP_SIZE groups of
        M = B // K sequences each.  Any remainder (B mod K) sequences receive a
        joint advantage of zero so the tensor shape is always (B,).

        Micro-advantage (Decoder credit assignment)
        -------------------------------------------
        Within each group k the M rewards are normalised independently:
            μ_k  = mean(R^{(k,*)})
            σ_k  = std(R^{(k,*)})
            Â_micro^{(k,m)} = (R^{(k,m)} - μ_k) / (σ_k + ε)

        Macro-advantage (Encoder credit assignment)
        -------------------------------------------
        The K group means are normalised against each other:
            μ_macro  = mean(μ_k for k in 0..K-1)
            σ_macro  = std(μ_k for k in 0..K-1)
            Â_macro^{(k)} = (μ_k - μ_macro) / (σ_macro + ε)

        Joint advantage
        ---------------
            Â_joint^{(k,m)} = Â_macro^{(k)} + Â_micro^{(k,m)}

        Lagrangian constraint costs and EOS penalties are then subtracted from the
        joint advantage exactly as in the previous ATLAS-GRPO formulation, so all
        constraint budget enforcement is preserved unchanged.

        Args:
            rewards:           Scalar reward tensor (B,) on CPU.
            constraint_costs:  Dict of constraint-name → cost tensor (B,) on CPU.
            eos_costs:         EOS-penalty tensor (B,) on CPU.
            importance_weights: Optional IS weight tensor (B,) on CPU.

        Returns:
            Tuple of (joint_advantages (B,), pivot_idx) where pivot_idx is the
            index of the rollout closest to the median macro-group mean (used by
            the caller to zero-out the pivot row in the loss).

        Time complexity:  O(B).
        Space complexity: O(B).
        """
        B = rewards.shape[0]
        K = GIGPO_LATENT_GROUP_SIZE
        M = max(B // K, 1)
        KM = K * M   # portion of the batch that fits evenly into K groups

        joint_adv = torch.zeros(B, dtype=torch.float32, device=rewards.device)

        if KM > 0 and KM <= B:
            group_rewards = rewards[:KM].reshape(K, M)   # (K, M)

            # ── Macro-advantage: Encoder credit assignment ────────────────────
            mu_k      = group_rewards.mean(dim=1)                           # (K,)
            mu_macro  = mu_k.mean()
            sig_macro = mu_k.std(unbiased=False).clamp(min=1e-8)
            macro_adv = (mu_k - mu_macro) / (sig_macro + 1e-8)             # (K,)

            # ── Micro-advantage: Decoder credit assignment ────────────────────
            mu_m  = group_rewards.mean(dim=1, keepdim=True)                 # (K, 1)
            sig_m = group_rewards.std(
                dim=1, unbiased=False, keepdim=True
            ).clamp(min=1e-8)                                               # (K, 1)
            micro_adv = (group_rewards - mu_m) / (sig_m + 1e-8)            # (K, M)

            # ── Additive combination ──────────────────────────────────────────
            joint_raw          = macro_adv.unsqueeze(1) + micro_adv        # (K, M)
            joint_adv[:KM]     = joint_raw.reshape(KM)

            # Pivot: sequence closest to the median macro group-mean, used by
            # compute_gigpo_loss to zero-out the pivot row (variance reduction).
            median_macro   = mu_k.median()
            deviations_mac = (mu_k - median_macro).abs()
            best_k         = int(deviations_mac.argmin().item())
            # Map the best group's first member to the flat batch index.
            pivot_idx = best_k * M
        else:
            # Degenerate case: fewer rollouts than K — fall back to median centering.
            median_reward = rewards.median()
            joint_adv     = rewards - median_reward
            deviations    = joint_adv.abs()
            pivot_idx     = int(deviations.argmin().item())

        # ── Lagrangian constraint penalties (preserved from ATLAS-GRPO) ───────
        for constraint_name, cost_tensor in constraint_costs.items():
            lambda_k      = self.lagrangian_multipliers.get(constraint_name, 0.0)
            centered_cost = cost_tensor - cost_tensor.median()
            joint_adv     = joint_adv - lambda_k * centered_cost

        centered_eos = eos_costs - eos_costs.median()
        joint_adv    = joint_adv - self.lagrangian_eos * centered_eos

        if importance_weights is not None:
            joint_adv = joint_adv * importance_weights

        joint_adv = torch.clamp(
            joint_adv, min=-ADVANTAGE_CLAMP_MAX, max=ADVANTAGE_CLAMP_MAX
        )
        return joint_adv, int(pivot_idx)

    def _update_lagrangian_multipliers(
        self,
        constraint_costs: Dict[str, torch.Tensor],
        eos_costs: torch.Tensor,
    ) -> None:
        """Dual-ascent update for Lagrangian constraint multipliers."""
        for constraint_name, cost_tensor in constraint_costs.items():
            empirical_rate = cost_tensor.mean().item()
            budget         = _CONSTRAINT_BUDGETS.get(constraint_name, 0.0)
            current_lambda = self.lagrangian_multipliers.get(constraint_name, 0.0)
            self.lagrangian_multipliers[constraint_name] = max(
                0.0,
                current_lambda + self.lagrangian_lr * (empirical_rate - budget),
            )
        self.lagrangian_eos = max(
            0.0,
            self.lagrangian_eos + self.lagrangian_lr * eos_costs.mean().item(),
        )

    def compute_gigpo_loss(
        self,
        student_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        pivot_idx: int = -1,
        n_replay: int = 0,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute Enhanced LAA-GiGPO loss with stability improvements.

        Improvements over baseline:
        1. Symmetric KL penalty (forward + reverse) for better regularization [[19-24]]
        2. Adaptive clipping bounds based on advantage sign [[4], [31]]
        3. Token-level advantage normalization for variance reduction [[13], [15]]
        4. Entropy-aware loss weighting for exploration maintenance [[1], [34]]
        5. Improved Carousel Alignment with gradient clipping [[29], [38]]
        6. Numerical stability enhancements throughout

        Args:
            student_log_probs: Current student per-token log-probs (B, S).
            old_log_probs:     Old (frozen) per-token log-probs (B, S).
            ref_log_probs:     Reference (teacher) per-token log-probs (B, S).
            advantages:        GiGPO joint-advantage per sequence (B,).
            mask:              Binary token mask; 1.0 at generated positions (B, S).
            pivot_idx:         Batch index of the pivot (median) sequence.
            n_replay:          Number of replay sequences at the END of the batch.

        Returns:
            Tuple of (scalar loss, mean_kl (float), mean_entropy_nats (float)).
        """
        # Numerical Stability: Cast to float32 early 
        student_lp_f32 = student_log_probs.float()
        old_lp_f32     = old_log_probs.float()
        ref_lp_f32     = ref_log_probs.float()
        mask_f32       = mask.float()
        eps            = 1e-8  # Numerical stability constant

        batch_size    = student_lp_f32.shape[0]
        seq_len       = student_lp_f32.shape[1]

        # Advantage Normalization: Token-level for finer credit assignment 
        # Normalize advantages per sequence, then expand to token level 
        adv_mean = (advantages * mask_f32.sum(dim=1)).sum() / mask_f32.sum().clamp(min=eps)
        adv_std  = torch.sqrt(
            ((advantages.unsqueeze(1) - adv_mean) ** 2 * mask_f32).sum() / 
            mask_f32.sum().clamp(min=eps) + eps
        )
        advantages_norm = (advantages - adv_mean) / adv_std.clamp(min=eps)
        adv           = advantages_norm.float().unsqueeze(1).expand_as(student_lp_f32)

        #  Policy Ratio with Improved Clipping 
        # Compute log-ratio first for numerical stability
        log_ratio   = student_lp_f32 - old_lp_f32
        log_ratio   = log_ratio.clamp(-5.0, 5.0)  # Prevent extreme ratios
        ratio       = torch.exp(log_ratio)
        
        # Adaptive clipping: asymmetric bounds based on advantage sign 
        pos_adv_mask = (adv > 0).float()
        neg_adv_mask = (adv <= 0).float()
        
        # Tighter clipping for positive advantages (conservative improvement)
        # Looser clipping for negative advantages (allow correction)
        clip_lo = self.epsilon_lo * (1.0 + 0.5 * neg_adv_mask)
        clip_hi = self.epsilon_hi * (1.0 - 0.3 * pos_adv_mask)
        
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_lo, 1.0 + clip_hi)
        
        # PPO surrogate loss with both clipped and unclipped terms
        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        policy_loss = -torch.min(surr1, surr2)

        # Enhanced KL Penalty: Symmetric KL for better regularization
        # Standard forward KL: KL(π_ref || π_θ) [[19-24]]
        diff_forward  = ref_lp_f32 - student_lp_f32
        kl_forward    = torch.exp(diff_forward) - diff_forward - 1.0
        
        # Reverse KL: KL(π_θ || π_ref) for symmetric regularization
        diff_reverse  = student_lp_f32 - ref_lp_f32
        kl_reverse    = torch.exp(diff_reverse) - diff_reverse - 1.0
        
        # Symmetric KL (Jeffrey's divergence) with tunable weighting
        kl_div = (1.0 - self.kl_symmetry_weight) * kl_forward + \
                 self.kl_symmetry_weight * kl_reverse
        
        # KL annealing: reduce penalty early in training for exploration
        kl_scale = min(1.0, self.global_step / self.kl_anneal_steps) if hasattr(self, 'global_step') else 1.0

        # Entropy Bonus: Encourage exploration 
        # Compute per-token entropy (negative log-prob weighted by probability)
        student_probs = torch.exp(student_lp_f32.clamp(-10.0, 0.0))  # Clamp for stability
        entropy = -student_probs * student_lp_f32  # H(π) = -Σ π log π
        entropy_bonus = -self.entropy_coef * entropy  # Negative because we maximize entropy

        # Combine Per-Token Loss 
        per_token_loss = (policy_loss + kl_scale * self.beta * kl_div + entropy_bonus) * mask_f32

        # Pivot Zeroing (Variance Reduction) 
        if 0 <= pivot_idx < batch_size:
            mask_pivot            = torch.ones_like(per_token_loss)
            mask_pivot[pivot_idx] = 0.0
            per_token_loss        = per_token_loss * mask_pivot

        # Loss Aggregation with Proper Token Counting 
        total_tokens = mask_f32.sum().clamp(min=eps)
        if 0 <= pivot_idx < batch_size:
            pivot_len    = mask_f32[pivot_idx].sum()
            total_tokens = (total_tokens - pivot_len).clamp(min=eps)

        total_loss = per_token_loss.sum() / total_tokens

        # Carousel Memory Alignment (Enhanced) 
        if n_replay > 0:
            replay_student_lp = student_lp_f32[-n_replay:]    # (n_replay, S)
            replay_old_lp     = old_lp_f32[-n_replay:]        # (n_replay, S)
            replay_mask       = mask_f32[-n_replay:]          # (n_replay, S)
            replay_tokens     = replay_mask.sum().clamp(min=eps)

            # Improved alignment: weighted Huber with decay based on sequence position
            alignment_diff = replay_student_lp - replay_old_lp
            alignment_huber = torch.where(
                alignment_diff.abs() < 1.0,
                0.5 * alignment_diff ** 2,
                1.0 * (alignment_diff.abs() - 0.5)
            )
            
            # Apply temporal decay: earlier tokens in replay get higher weight
            token_positions = torch.arange(seq_len, device=alignment_huber.device).float() / seq_len
            temporal_weight = 1.0 - 0.5 * token_positions  # Decay from 1.0 to 0.5
            alignment_huber = alignment_huber * temporal_weight.unsqueeze(0)
            
            carousel_loss = (alignment_huber * replay_mask).sum() / replay_tokens
            
            # Gradient clipping for carousel loss to prevent instability 
            if self.carousel_grad_clip > 0:
                carousel_loss = torch.clamp(
                    carousel_loss, 
                    -self.carousel_grad_clip, 
                    self.carousel_grad_clip
                )
            
            total_loss = total_loss + GIGPO_CAROUSEL_ALIGN_WEIGHT * carousel_loss

        # Metrics Computation 
        mean_entropy_nats = (
            (-student_probs * student_lp_f32 * mask_f32).sum() / 
            mask_f32.sum().clamp(min=eps)
        )
        mean_kl = (kl_div * mask_f32).sum() / mask_f32.sum().clamp(min=eps)
        
        #  Final Loss Clipping (Training Stability)
        total_loss = torch.clamp(total_loss, -10.0, 10.0)  # Prevent explosion

        return total_loss.to(student_log_probs.dtype), mean_kl.item(), mean_entropy_nats.item()

    # ILP helpers 

    def _violates_ilp_constraints(self, state_tensor: torch.Tensor) -> bool:
        """Return True if state_tensor is too far from any known positive example."""
        if self.ilp_positive_examples:
            # FIX: Ensure all tensors are on CPU before concatenation.
            # ilp_positive_examples may contain a mix of CPU tensors (from current training)
            # and CUDA tensors (restored from checkpoint), which causes torch.cat to crash.
            known_valid = torch.cat([t.cpu() for t in self.ilp_positive_examples], dim=0)

            query = state_tensor.cpu()
            if query.dim() == 1:
                query = query.unsqueeze(0)

            # Both query and known_valid are now guaranteed to be on CPU.
            min_dist = torch.cdist(query, known_valid).min()
            if min_dist > 1.5:
                return True
        return False

    def _mcts_generate_program(
        self,
        template: torch.Tensor,
        size: int,
        pos_examples: List[torch.Tensor],
        neg_examples: List[torch.Tensor],
        local_const: Set[str],
        num_simulations: int = 50
    ) -> Optional[List[torch.Tensor]]:
        """Use MCTS to find the optimal sub-sequence of length `size`."""
        T = template.shape[0]
        if T < size:
            return None

        root = ILPMCTSNode(indices=[], target_len=size, max_idx=T)

        for _ in range(num_simulations):
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child()
            if not node.is_terminal():
                node = node.expand()

            rollout_indices = node.indices.copy()
            while len(rollout_indices) < size:
                last_idx      = rollout_indices[-1] if rollout_indices else -1
                needed        = size - len(rollout_indices) - 1
                possible_next = list(range(last_idx + 1, T - needed))
                if not possible_next:
                    break
                rollout_indices.append(random.choice(possible_next))

            if len(rollout_indices) < size:
                continue

            candidate_program = [template[i].unsqueeze(0) for i in rollout_indices]
            prog_val          = sum(t.sum().item() for t in candidate_program)
            prog_hash         = hash(str(prog_val))

            is_pruned = any(
                self._violates_ilp_constraints(state) for state in candidate_program
            ) or (
                f"Prune_Spec_{prog_hash}" in local_const
                or f"Prune_Gen_{prog_hash}" in local_const
            )

            if is_pruned:
                reward = -1.0
            else:
                tp, fp, fn, inconsistent = self._test_program(
                    candidate_program, pos_examples, neg_examples
                )
                tp_rate = tp / max(len(pos_examples), 1)
                fp_rate = fp / max(len(neg_examples), 1)
                reward  = tp_rate - (2.0 * fp_rate)
                if inconsistent:
                    reward -= 0.5

            while node is not None:
                node.visits += 1
                node.value  += reward
                node         = node.parent

        best_node = root
        while not best_node.is_terminal():
            if not best_node.children:
                return None
            best_node = best_node.best_child(c_param=0.0)

        return [template[i].unsqueeze(0) for i in best_node.indices]

    def _generate_simulated_program(
        self,
        size: int,
        global_const: Set[str],
        local_const: Set[str],
    ) -> Optional[List[torch.Tensor]]:
        """Sub-sample a positive trajectory to generate an ILP program candidate."""
        with torch.no_grad():
            try:
                valid    = [t for t in self.ilp_positive_examples if t.shape[0] >= size]
                if not valid:
                    return None
                template = random.choice(valid)
                indices  = sorted(random.sample(range(template.shape[0]), size))
                program  = [template[i].unsqueeze(0) for i in indices]
                for state in program:
                    if self._violates_ilp_constraints(state):
                        return None
                prog_val  = sum(t.sum().item() for t in program)
                prog_hash = hash(str(prog_val))
                if (
                    f"Prune_Spec_{prog_hash}" in local_const
                    or f"Prune_Gen_{prog_hash}" in local_const
                ):
                    return None
                return program
            except (IndexError, RuntimeError, ValueError):
                return None

    def _is_covered(
        self, program: List[torch.Tensor], trajectory: torch.Tensor
    ) -> bool:
        """Check whether a trajectory covers all program clauses in order."""
        prog_idx = traj_idx = 0
        while prog_idx < len(program) and traj_idx < trajectory.shape[0]:
            dist = torch.norm(
                trajectory[traj_idx].cpu() - program[prog_idx].cpu().squeeze()
            )
            if dist < 0.5:
                prog_idx += 1
            traj_idx += 1
        return prog_idx == len(program)

    def _test_program(
        self,
        program: List[torch.Tensor],
        pos_examples: List[torch.Tensor],
        neg_examples: List[torch.Tensor],
    ) -> Tuple[int, int, int, bool]:
        """Evaluate a program against positive and negative examples."""
        true_positives  = sum(1 for ex in pos_examples if self._is_covered(program, ex))
        false_positives = sum(1 for ex in neg_examples if self._is_covered(program, ex))
        false_negatives = len(pos_examples) - true_positives
        return true_positives, false_positives, false_negatives, false_positives > 0

    def _simulate_combiner_maxsat(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select the best candidate program by weighted MaxSAT heuristic."""
        best       = candidates[0] if candidates else {}
        best_score = float("inf")
        for cand in candidates:
            score = cand['fp'] + cand['fn'] + 0.1 * cand['size']
            cand['score'] = score
            if score < best_score:
                best_score = score
                best       = cand
        return best

    def _extract_and_broadcast_metarules(
        self, h_best: List[torch.Tensor]
    ) -> None:
        """Register the best ILP program as a named metarule."""
        prog_val = sum(t.sum().item() for t in h_best)
        rule_sig = f"Rule_{hash(str(prog_val))}"
        if rule_sig not in self.global_knowledge_base:
            self.global_knowledge_base.add(rule_sig)
            logging.getLogger("RL-ILP-Agent").info(
                "Extracted new Metarule: %s (Size: %d)", rule_sig, len(h_best)
            )

    def _run_ilp_meta_planner(self) -> None:
        """Run ILP meta-planning to extract symbolic rules from recent examples."""
        if len(self.ilp_positive_examples) < 2:
            return

        candidates: List[Dict[str, Any]]    = []
        score_best: float                   = float("inf")
        h_best: Optional[List[torch.Tensor]] = None

        current_max_len = (
            max(t.shape[0] for t in self.ilp_positive_examples)
            if self.ilp_positive_examples
            else 0
        )
        max_possible_k = min(self.max_program_size, current_max_len)

        for k in range(1, max_possible_k + 1):
            local_constraints: Set[str] = set()

            if self.use_mcts:
                num_templates  = min(3, len(self.ilp_positive_examples))
                sampled        = random.sample(self.ilp_positive_examples, num_templates)

                for template in sampled:
                    candidate = self._mcts_generate_program(
                        template=template,
                        size=k,
                        pos_examples=self.ilp_positive_examples,
                        neg_examples=self.ilp_negative_examples,
                        local_const=local_constraints,
                        num_simulations=40
                    )
                    if candidate is None:
                        continue

                    tp, fp, fn, inconsistent = self._test_program(
                        candidate, self.ilp_positive_examples, self.ilp_negative_examples
                    )
                    if tp == len(self.ilp_positive_examples) and fp == 0 and not inconsistent:
                        h_best     = candidate
                        score_best = 0.0
                        break

                    prog_val  = sum(t.sum().item() for t in candidate)
                    prog_hash = hash(str(prog_val))
                    if tp == 0:
                        local_constraints.add(f"Prune_Spec_{prog_hash}")
                    elif inconsistent:
                        local_constraints.add(f"Prune_Gen_{prog_hash}")
                    if not inconsistent and tp > 0:
                        candidates.append(
                            {"program": candidate, "tp": tp, "fp": fp, "fn": fn, "size": k}
                        )
                if score_best == 0.0:
                    break
            else:
                for _ in range(5 * k):
                    candidate = self._generate_simulated_program(
                        k, self.global_knowledge_base, local_constraints
                    )
                    if candidate is None:
                        continue

                    tp, fp, fn, inconsistent = self._test_program(
                        candidate, self.ilp_positive_examples, self.ilp_negative_examples
                    )
                    if tp == len(self.ilp_positive_examples) and fp == 0 and not inconsistent:
                        h_best     = candidate
                        score_best = 0.0
                        break

                    prog_val  = sum(t.sum().item() for t in candidate)
                    prog_hash = hash(str(prog_val))
                    if tp == 0:
                        local_constraints.add(f"Prune_Spec_{prog_hash}")
                    elif inconsistent:
                        local_constraints.add(f"Prune_Gen_{prog_hash}")
                    if not inconsistent and tp > 0:
                        candidates.append(
                            {"program": candidate, "tp": tp, "fp": fp, "fn": fn, "size": k}
                        )

                    if score_best == 0.0:
                        break

        if h_best is None and candidates:
            best_cand = self._simulate_combiner_maxsat(candidates)
            if best_cand['score'] < score_best:
                h_best = best_cand['program']

        if h_best is not None:
            self.current_logical_plan = h_best
            self._extract_and_broadcast_metarules(h_best)
            self.ilp_positive_examples.clear()
            self.ilp_negative_examples.clear()

    # Replay padding helper 

    def _pad_replay_to_online(
        self, replay_ids: torch.Tensor, target_seq_len: int
    ) -> torch.Tensor:
        """Pad or truncate a replay trajectory to match the online sequence length."""
        current_len = replay_ids.shape[1]
        pad_token   = self.teacher.tokenizer.pad_token_id
        if current_len >= target_seq_len:
            return replay_ids[:, :target_seq_len]
        padding = torch.full(
            (1, target_seq_len - current_len),
            pad_token,
            dtype=replay_ids.dtype,
            device=replay_ids.device,
        )
        return torch.cat([replay_ids, padding], dim=1)

    # Main training step 
    def train_step(self, prompt_text: str) -> Dict[str, float]:
        """Execute one LAA-GiGPO training micro-step.

        Args:
            prompt_text: Plain content string (no chat-template markers).
                         The chat template is applied exactly once inside this method.

        Returns:
            Dict of scalar training metrics.
        """
        self.total_steps  += 1
        self._accum_step  += 1
        self._micro_step  += 1
        is_opt_step = (self._micro_step % self._GRAD_ACCUM_STEPS == 0)

        # ── Tokenise 
        # Apply chat template - called here and ONLY here
        # prompt_text is a plain string; do not embed <|user|> markers in prompts.
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = self.teacher.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        tokenised = self.teacher.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.SEQ_LEN,
            padding=True,
            return_attention_mask=True,
        ).to(self.device)
        input_ids  = tokenised.input_ids
        prompt_len = input_ids.shape[1]

        #  Teacher demonstration 
        with torch.no_grad():
            expert_out = self.teacher.generate(
                input_ids,
                attention_mask=tokenised.attention_mask,
                max_new_tokens=self.t_max,
                pad_token_id=self.teacher.tokenizer.pad_token_id,
            )

        #  Supervised CE loss (student forward with grad) 
        self.student.train()
        with self._amp_ctx:
            supervised_student_out = self.student(
                expert_out, teacher_hidden_states=None
            )
            supervised_logits = supervised_student_out["logits"]
            aux_loss = torch.clamp(
                supervised_student_out.get("aux_loss", expert_out.new_zeros(())),
                max=10.0,
            )
            supervised_ce_loss = self.compute_supervised_loss(
                supervised_logits, expert_out, prompt_len
            )

        # Hidden-state alignment loss 
        hidden_loss = expert_out.new_zeros(())
        if HIDDEN_LOSS_WEIGHT > 0:
            with torch.no_grad():
                teacher_full_out   = self.teacher(expert_out)
                teacher_hidden_all = teacher_full_out["hidden_states"]

            student_hidden_list = supervised_student_out["hidden_states"]
            t_stride = max(Config.get_teacher_num_layers() // Config.NUM_LAYERS, 1)
            with self._amp_ctx:
                proj_list   = []
                target_list = []
                for s_idx, student_hidden in enumerate(student_hidden_list):
                    t_idx = min(1 + s_idx * t_stride, len(teacher_hidden_all) - 1)
                    proj_list.append(self.student.project_state(s_idx, student_hidden))
                    target_list.append(teacher_hidden_all[t_idx].detach())
                proj_stack   = torch.stack(proj_list).float()
                target_stack = torch.stack(target_list).float()
                hidden_loss  = F.mse_loss(proj_stack, target_stack) * HIDDEN_LOSS_SCALE

        # Online rollouts 
        online_group_size = self.G + 1 if self.G % 2 == 0 else self.G
        self.student.eval()
        with torch.no_grad():
            online_rollouts = self.student.generate(
                input_ids.repeat(online_group_size, 1),
                max_new_tokens=self.t_max,
                do_sample=True,
                temperature=ONLINE_GENERATION_TEMPERATURE,
            )
            online_seq_len      = online_rollouts.shape[1]
            ref_teacher_out     = self.teacher(online_rollouts)
            ref_logits_online   = ref_teacher_out["logits"]
            online_rewards      = self.compute_soft_reward(
                ref_logits_online, online_rollouts, prompt_len
            )
            old_student_out     = self.student(online_rollouts)
            old_log_probs_online = self.get_log_probs(
                old_student_out["logits"], online_rollouts
            ).detach()
            ref_log_probs_online = self.get_log_probs(
                ref_logits_online, online_rollouts
            ).detach()

        # Replay mixing 
        replay_count   = int(np.ceil(self.replay_alpha * online_group_size))
        replay_entries = self.experience_buffer.sample(replay_count)
        
        # Track valid lengths for masking to avoid padding explosion
        replay_valid_lengths = []

        if replay_entries:
            replay_ids_list      = []
            replay_old_lp_list   = []
            replay_rewards_list  = []
            for entry in replay_entries:
                # FIX: Ensure inputs are on self.device before processing.
                # Checkpoints might load buffer tensors to GPU, while fresh insertions are CPU.
                # Standardizing on self.device here prevents mixed-device errors in torch.cat.
                raw_ids = entry["trajectory_ids"].to(self.device)
                raw_lp  = entry["source_log_probs"].to(self.device)

                padded_ids = self._pad_replay_to_online(
                    raw_ids.unsqueeze(0), online_seq_len
                )
                padded_lp  = self._pad_replay_to_online(
                    raw_lp.unsqueeze(0), online_seq_len
                )
                replay_ids_list.append(padded_ids)
                replay_old_lp_list.append(padded_lp)
                replay_rewards_list.append(entry["reward"])
                
                # Store valid length for masking later
                valid_len = raw_ids.shape[0]
                replay_valid_lengths.append(min(valid_len, online_seq_len))

            replay_ids_tensor     = torch.cat(replay_ids_list, dim=0)
            replay_old_lp_tensor  = torch.cat(replay_old_lp_list, dim=0)
            replay_rewards_tensor = torch.tensor(replay_rewards_list, device=self.device)

            with torch.no_grad():
                ref_teacher_replay   = self.teacher(replay_ids_tensor)
                ref_log_probs_replay = self.get_log_probs(
                    ref_teacher_replay["logits"], replay_ids_tensor
                ).detach()

            aug_rollouts = torch.cat([online_rollouts, replay_ids_tensor], dim=0)
            aug_old_lp   = torch.cat([old_log_probs_online, replay_old_lp_tensor], dim=0)
            aug_ref_lp   = torch.cat([ref_log_probs_online, ref_log_probs_replay], dim=0)
            aug_rewards  = torch.cat([online_rewards, replay_rewards_tensor], dim=0)
            n_replay     = len(replay_entries)
            importance_weights = torch.ones(
                online_group_size + n_replay, device=self.device
            )
        else:
            aug_rollouts       = online_rollouts
            aug_old_lp         = old_log_probs_online
            aug_ref_lp         = ref_log_probs_online
            aug_rewards        = online_rewards
            n_replay           = 0
            importance_weights = None

        aug_total = online_group_size + n_replay

        # Reward shaping & completion scoring 
        with torch.no_grad():
            gen_seqs      = aug_rollouts
            decoded_batch = self.teacher.tokenizer.batch_decode(
                gen_seqs, skip_special_tokens=True
            )
            generated_token_counts = [
                (gen_seqs[i] != self.teacher.tokenizer.pad_token_id).sum().item()
                for i in range(aug_total)
            ]
            penalty_tensor = torch.zeros_like(aug_rewards)
            for idx, text in enumerate(decoded_batch):
                stripped = text.strip()
                if not stripped:
                    penalty_tensor[idx] -= EMPTY_TEXT_PENALTY
                elif len(set(text)) < 5 and len(text) > 15:
                    penalty_tensor[idx] -= REPETITION_PENALTY
                else:
                    # Lower word-loop threshold for repetition detection
                    words = text.lower().split()
                    if len(words) > 10:
                        counts: Dict[str, int] = {}
                        max_w_count = 0
                        for w in words:
                            c = counts.get(w, 0) + 1
                            counts[w] = c
                            if c > max_w_count:
                                max_w_count = c
                        if max_w_count / len(words) > 0.125:
                            penalty_tensor[idx] -= REPETITION_PENALTY

            aug_rewards       = aug_rewards + penalty_tensor
            completion_scores = self._compute_completion_scores(
                decoded_batch, generated_token_counts
            )
            constraint_costs  = self._compute_constraint_costs(decoded_batch)
            eos_costs         = self._compute_eos_costs(
                completion_scores, aug_rollouts, prompt_len
            )

        # Advantage computation 
        all_rewards_equal = (aug_rewards.max() - aug_rewards.min()).abs() < self.eps_thresh
        if all_rewards_equal and len(self.experience_buffer) == 0:
            advantages = torch.zeros(aug_total, device=self.device)
            pivot_idx  = 0
        else:
            advantages, pivot_idx = self._compute_gigpo_advantages(
                rewards=aug_rewards.cpu(),
                constraint_costs={k: v.cpu() for k, v in constraint_costs.items()},
                eos_costs=eos_costs.cpu(),
                importance_weights=(
                    importance_weights.cpu()
                    if importance_weights is not None
                    else None
                ),
            )
            advantages = advantages.to(self.device)

        reward_mean = aug_rewards.mean().item()
        reward_std  = aug_rewards.std().item()

        if self._accum_step % 25 == 0:
            gen_only = aug_rollouts[0]
            decoded  = self.teacher.tokenizer.decode(gen_only, skip_special_tokens=True)
            logging.getLogger("RL-ILP-Agent").info(
                " Step %d  Gen (R=%.2f, S_comp=%.2f): '%s'",
                self.total_steps, aug_rewards[0].item(),
                completion_scores[0].item(), decoded,
            )
            _p1 = predicate_has_non_trivial_content(decoded_batch[0])
            _p2 = predicate_has_minimum_generation_length(generated_token_counts[0])
            _p3 = predicate_has_structural_completion(decoded_batch[0])
            _eos_in_gen = int(
                (aug_rollouts[0] == self.teacher.tokenizer.eos_token_id).any().item()
            )
            print(
                f"\n Step {self.total_steps} "
                f"Gen (Reward: {aug_rewards[0].item():.2f}): '{decoded}'"
            )
            print(
                f"   predicates → non_trivial={int(_p1)} "
                f"min_len={int(_p2)} structural={int(_p3)} "
                f"| S_comp={completion_scores[0].item():.2f} "
                f"| eos_in_gen={_eos_in_gen} "
                f"| λ_eos={self.lagrangian_eos:.4f}"
            )

        # ILP meta-planner update 
        decoded_online   = self.teacher.tokenizer.batch_decode(
            online_rollouts.cpu(), skip_special_tokens=True
        )
        semantic_vectors = self.semantic_encoder(decoded_online)
        batch_avg_reward = online_rewards.mean().item()
        for i in range(online_group_size):
            vec = semantic_vectors[i].detach().cpu().unsqueeze(0)
            if online_rewards[i].item() >= batch_avg_reward:
                self.ilp_positive_examples.append(vec)
            else:
                self.ilp_negative_examples.append(vec)
        self._run_ilp_meta_planner()

        # LAA-GiGPO policy update 
        # Phase A (decoder): PPO-clip on micro+macro joint advantages.
        # Phase B (encoder): handled implicitly via the same backward pass.
        self.student.train()
        rollouts_gpu = aug_rollouts.clone()
        old_lp_gpu   = aug_old_lp.to(self.device).detach()
        ref_lp_gpu   = aug_ref_lp.to(self.device).detach()

        with self._amp_ctx:
            current_student_out = self.student(rollouts_gpu)
            current_logits      = current_student_out["logits"]
            current_log_probs   = self.get_log_probs(current_logits, rollouts_gpu)
            grpo_aux = current_student_out.get(
                "aux_loss", rollouts_gpu.new_zeros(())
            )
            aux_loss = aux_loss + torch.clamp(grpo_aux, max=10.0)
            
            # Construct strict mask to prevent padding explosion.
            mask = torch.zeros_like(rollouts_gpu, dtype=torch.float32)
            
            # 1. Mask for online rollouts: valid from prompt_len onwards.
            mask[:online_group_size, prompt_len:] = 1.0
            
            # 2. Mask for replay rollouts: valid from prompt_len up to valid_len.
            # Replay padding contains large integer tokens (pad_id) which 
            # resulted in 'source_log_probs' containing large values.
            # Masking prevents these from entering the loss.
            if n_replay > 0:
                for r_i, v_len in enumerate(replay_valid_lengths):
                    batch_idx = online_group_size + r_i
                    # Only mask if valid length extends past prompt
                    if v_len > prompt_len:
                        mask[batch_idx, prompt_len:v_len] = 1.0

            grpo_loss, kl_metric, entropy_metric = self.compute_gigpo_loss(
                current_log_probs, old_lp_gpu, ref_lp_gpu,
                advantages, mask, pivot_idx=pivot_idx, n_replay=n_replay,
            )

        # Adaptive entropy bonus
        entropy_alpha = self.entropy_controller.step(entropy_metric)
        entropy_bonus = -entropy_alpha * entropy_metric

        grpo_weight = get_rl_grpo_weight(self.total_steps)
        total_loss  = (
            CE_LOSS_WEIGHT       * supervised_ce_loss
            + grpo_weight        * grpo_loss
            + HIDDEN_LOSS_WEIGHT * hidden_loss
            + aux_loss
            + entropy_bonus
        ) / self._GRAD_ACCUM_STEPS

        self.scaler.scale(total_loss).backward()

        if is_opt_step:
            self.scaler.unscale_(self.opt_student)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.opt_student)
            self.scaler.update()
            self.opt_student.zero_grad(set_to_none=True)
            self.scheduler.step()
            self._micro_step = 0

        if (
            hasattr(self, "ema")
            and self.ema is not None
            and self.total_steps % self._EMA_UPDATE_FREQ == 0
        ):
            self.ema.update(self.student)

        self._update_lagrangian_multipliers(
            constraint_costs={k: v.cpu() for k, v in constraint_costs.items()},
            eos_costs=eos_costs.cpu(),
        )

        with torch.no_grad():
            median_reward = online_rewards.median().item()
            for i in range(online_group_size):
                if (
                    online_rewards[i].item() > median_reward
                    and completion_scores[i].item() > 0.0
                ):
                    self.experience_buffer.insert(
                        trajectory_ids=online_rollouts[i],
                        prompt_length=prompt_len,
                        reward=online_rewards[i].item(),
                        source_log_probs=old_log_probs_online[i],
                        timestamp=self.total_steps,
                    )

        if self.total_steps % 50 == 0:
            self.save_checkpoint(f"deepmime_step_{self.total_steps}.pth")

        return {
            "total_loss":         total_loss.item() * self._GRAD_ACCUM_STEPS,
            "supervised_ce_loss": supervised_ce_loss.item(),
            "grpo_loss":          grpo_loss.item(),
            "hidden_loss":        hidden_loss.item(),
            "aux_loss":           aux_loss.item(),
            "reward_mean":        reward_mean,
            "reward_std":         reward_std,
            "kl":                 kl_metric,
            "entropy":            entropy_metric,
            "entropy_alpha":      entropy_alpha,
            "grpo_weight":        grpo_weight,
            "lr":                 self.opt_student.param_groups[0]["lr"],
            "lambda_eos":         self.lagrangian_eos,
            "lambda_format":      self.lagrangian_multipliers.get("format_compliance", 0.0),
            "lambda_repetition":  self.lagrangian_multipliers.get("repetition_penalty", 0.0),
            "replay_buffer_size": float(len(self.experience_buffer)),
            "completion_score":   completion_scores.mean().item(),
            "eos_cost_mean":      eos_costs.mean().item(),
            "pivot_idx":          float(pivot_idx),
            "n_online":           float(online_group_size),
            "n_replay":           float(n_replay),
        }


# ==============================================================================
# 9. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    START_FROM_SCRATCH: str  = "yes"
    TARGET_TOTAL_STEPS: int  = 3250
    CHECKPOINT_FILENAME: str = "deepmime_v2_latest.pth"
    CHECKPOINT_DIR: Path     = Path("checkpoints")

    print(get_syllabus_summary())

    if START_FROM_SCRATCH.lower() == "yes":
        if CHECKPOINT_DIR.exists() and CHECKPOINT_DIR.is_dir():
            shutil.rmtree(CHECKPOINT_DIR)
            print(
                f"\n[RESET] Deleted '{CHECKPOINT_DIR}' directory. "
                f"Training will start from scratch.\n"
            )
    else:
        print(
            f"\n[RESUME] Keeping '{CHECKPOINT_DIR}' to resume "
            f"training if possible.\n"
        )

    try:
        trainer = DeepMimeTrainer(total_training_steps=TARGET_TOTAL_STEPS)
        trainer.load_checkpoint(CHECKPOINT_FILENAME, load_ema=False)
        logging.getLogger("RL-ILP-Agent").info(
            "Training loop: Steps %d → %d",
            trainer.total_steps, TARGET_TOTAL_STEPS,
        )
        start_step = trainer.total_steps
        for step_idx in range(start_step, TARGET_TOTAL_STEPS):
            # get_prompt() returns a plain content string; train_step() wraps it
            # in a chat-template message internally — exactly once.
            prompt_text = get_prompt(step_idx)
            tier_label  = get_tier_name(step_idx)
            metrics     = trainer.train_step(prompt_text)
            print(
                f"Step {step_idx + 1:>5d}/{TARGET_TOTAL_STEPS} "
                f"[{tier_label:4s}] | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"CE: {metrics['supervised_ce_loss']:.4f} | "
                f"GRPO: {metrics['grpo_loss']:.4f} | "
                f"R: {metrics['reward_mean']:.4f} | "
                f"Ent: {metrics.get('entropy', 0.0):.4f} | "
                f"λ_eos: {metrics['lambda_eos']:.3f} | "
                f"S_comp: {metrics['completion_score']:.2f} | "
                f"Buf: {int(metrics['replay_buffer_size']):>4d} | "
                f"LR: {metrics['lr']:.2e}"
            )
            if (step_idx + 1) % 50 == 0:
                trainer.save_checkpoint(CHECKPOINT_FILENAME)

        trainer.save_checkpoint("deepmime_v2_final.pth")
        logging.getLogger("RL-ILP-Agent").info("Testing with EMA weights ...")

        if hasattr(trainer, "ema") and trainer.ema is not None:
            if hasattr(trainer.ema, "apply_shadow"):
                trainer.ema.apply_shadow(trainer.student)
            elif hasattr(trainer.ema, "module"):
                trainer.student = trainer.ema.module

            trainer.student.eval()
            test_prompt = "Write a function to calculate Fibonacci numbers in Python."
            msgs   = [{"role": "user", "content": test_prompt}]
            inputs = trainer.teacher.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            ).to(trainer.device)
            with torch.no_grad():
                output_ids = trainer.student.generate(
                    inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.875,
                    eos_token_id=trainer.teacher.tokenizer.eos_token_id,
                )
                generated_text = trainer.teacher.tokenizer.decode(
                    output_ids[0][inputs.shape[1]:],
                    skip_special_tokens=True,
                )
                print("\n" + "=" * 60)
                print("TEST GENERATION RESULTS (EMA weights)")
                print("=" * 60)
                print(f"Input Prompt:\n{test_prompt}\n")
                print(f"Student Generation:\n{generated_text}")
                print("=" * 60)

    except OSError as exc:
        print(f"Could not run training. Details: {exc}")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving emergency checkpoint ...")
        if "trainer" in locals():
            trainer.save_checkpoint("deepmime_v2_interrupted.pth")


# Alhamdulillahi Rabbil Alamin