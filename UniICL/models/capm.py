"""
Context-Adaptive Prototype Modulator (CAPM)

A lightweight plug-and-play module for unified understanding and generation
via in-context learning.

Architecture (3-stage pipeline):

  Stage 1 — Per-demo extraction (CapmProber → FacetPooler → OperatorMapper)
    CapmProber:      Segment-aware cross-attention with P learnable probes.
                      → c_in (User CLS), c_out (Assistant CLS), facets (K fine-grained)
    FacetPooler:      g = mean(RMSNorm(facets))  — pooled facet summary
    OperatorMapper:   Dictionary low-rank operator conditioned on (c_in, c_out, g).
                      Shared bases U_base, V_base with sample-wise scales.
                      z = g + U·(α ⊙ Vᵀ·g)  — near-identity at init via op_gain.

  Stage 1.5 — Cross-demo interaction (DemoInteraction)    ★ NEW
    Self-attention across z tokens of all N demos.
    Lets demos exchange complementary/redundant signals before bank assembly.
    Zero-init output projection → identity at init.

  Stage 2 — Bank assembly + adaptive cosine routing (TokenCalibrator → PatternAligner)
    Bank assembly:    Per demo [z, c_in, c_out, facets] → (B, S, d_p), S = N·(K+3).
    TokenCalibrator:  Type-wise affine (4 types: z/c_in/c_out/facets), γ=1 β=0 init.
    PatternAligner:   Dense cosine routing with **operator-conditioned** temperature τ.
                      τ = 0.05 + 0.7*sigmoid(τ_logit + 0.25*tanh(MLP(z_pool)))  ★ sigmoid
                      q = L2Norm(W_bridge(RMSNorm(H_in))), k = L2Norm(B_cal)
                      C = softmax(q·kᵀ / τ) @ B_cal

  Stage 3 — Per-layer injection (CapmGate)
    Gate:  mask = σ(MLP([LN(H_in); C]))
    Apply: Y' = Y ⊙ mask  (identity-initialized, preserves pretrained backbone)

Key novelty:
  1. Cross-demo interaction: first to model inter-demo relationships in ICL,
     enabling complementary enhancement and redundancy suppression.
  2. Operator-conditioned routing temperature: task-adaptive routing sharpness
     inferred from the demo-set representation, not a global hyperparameter.

Author: UniICL Team
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CapmConfig:
    """Configuration for the Context-Adaptive Prototype Modulator."""
    # Dimensions
    d_backbone: int = 3584      # Backbone hidden dimension (Qwen2.5-7B)
    d_capm: int = 768          # CAPM internal dimension

    # Architecture
    num_probes: int = 32        # P: total learnable probes (K = P-2 facets)
    cross_attn_heads: int = 8   # Heads for CapmProber cross-attention
    operator_rank: int = 64     # r: rank of low-rank operator
    op_gain: float = 0.1        # Gain multiplier for operator (near-identity init)

    # Injection
    num_inject_layers: int = 28 # Number of backbone layers to inject

    # Training
    dropout: float = 0.0
    detach_operator_from_facets: bool = False  # Optionally detach g from facets
    # Routing scope:
    # - False: query-only modulation (default, current behavior)
    # - True: route + gate both demo and query tokens when demo bank exists
    apply_to_demo_tokens: bool = False

    # Ablation: gate-only mode
    # When True, skip encoder/routing entirely, only instantiate CapmGate modules.
    # Gates receive zero context (C=0), modulation depends solely on H_in.
    # This isolates the contribution of the gating mechanism from demo-aware routing.
    gate_only: bool = False

    # Inference-time component ablation. The default keeps the original CAPM path.
    # Supported values:
    # - none: original implementation
    # - no_adaptive_routing: replace token-wise routing with mean-pooled bank context
    # - no_decoupled_encoding: use one shared demo representation for all bank slots
    # - no_low_rank_transformation: bypass the low-rank operator delta, z = g
    ablation_mode: str = "none"

    # Optional fixed routing temperature for inference-time tau ablation.
    # When set, this overrides the adaptive tau path in PatternAligner.
    fixed_tau: Optional[float] = None

# ============================================================================
# Building blocks
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen-style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ============================================================================
# Stage 1: Per-demo extraction
# ============================================================================


class CapmProber(nn.Module):
    """
    Cross-attention with P learnable probes over demo embeddings.

    Uses hard segment masking:
    - CLSin  (probe 0): attends only to User tokens (segment_id == 0)
    - CLSout (probe 1): attends only to Assistant tokens (segment_id == 1)
    - Facets (probes 2..P-1): attend to the full sequence

    Input:  demo_embeds (B, L_i, d_backbone), segment_ids (B, L_i)
    Output: c_in (B, d_capm), c_out (B, d_capm), facets (B, K, d_capm)
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.d_backbone = config.d_backbone
        self.d_capm = config.d_capm
        self.num_heads = config.cross_attn_heads
        self.head_dim = config.d_capm // config.cross_attn_heads
        self.num_probes = config.num_probes
        self.num_facets = config.num_probes - 2  # K = P - 2

        # Input projection: d_backbone -> d_capm
        self.input_proj = nn.Linear(config.d_backbone, config.d_capm, bias=False)

        # Learnable probes in CAPM space: (P, d_capm)
        self.probes = nn.Parameter(torch.randn(self.num_probes, config.d_capm) * 0.02)

        # Cross-attention projections (all in d_capm space)
        self.q_proj = nn.Linear(config.d_capm, config.d_capm, bias=False)
        self.k_proj = nn.Linear(config.d_capm, config.d_capm, bias=False)
        self.v_proj = nn.Linear(config.d_capm, config.d_capm, bias=False)
        self.o_proj = nn.Linear(config.d_capm, config.d_capm, bias=False)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        demo_embeds: torch.Tensor,           # (B, L_i, d_backbone)
        segment_ids: torch.Tensor,           # (B, L_i) - 0 for User, 1 for Assistant
        attention_mask: Optional[torch.Tensor] = None  # (B, L_i)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            c_in: CLSin representation (B, d_capm)
            c_out: CLSout representation (B, d_capm)
            facets: Facet representations (B, K, d_capm)
        """
        # Safety check: clean NaNs in input embeddings
        if torch.isnan(demo_embeds).any() or torch.isinf(demo_embeds).any():
             demo_embeds = torch.nan_to_num(demo_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

        B, L_i, _ = demo_embeds.shape
        P = self.num_probes

        # Project to CAPM space
        demo_embeds = self.input_proj(demo_embeds)  # (B, L_i, d_capm)
        d = self.d_capm

        # Expand probes for batch
        probes = self.probes.unsqueeze(0).expand(B, -1, -1)  # (B, P, d_capm)

        # Q from probes, K/V from demo
        Q = self.q_proj(probes)       # (B, P, d_capm)
        K = self.k_proj(demo_embeds)  # (B, L_i, d_capm)
        V = self.v_proj(demo_embeds)  # (B, L_i, d_capm)

        # Reshape for multi-head attention
        Q = Q.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_i, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_i, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Segment mask
        segment_mask = self._build_segment_mask(segment_ids, P, B, L_i, demo_embeds.device)
        segment_mask = segment_mask.unsqueeze(1)  # (B, 1, P, L_i)
        attn_scores = attn_scores.masked_fill(segment_mask == 0, float('-inf'))

        # Padding mask
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask == 0, float('-inf'))

        # Handle edge case: if all positions are -inf for a probe, softmax gives NaN
        # Replace -inf rows with zeros before softmax to avoid NaN
        all_inf_mask = torch.all(attn_scores == float('-inf'), dim=-1, keepdim=True)  # (B, H, P, 1)
        attn_scores = attn_scores.masked_fill(all_inf_mask, 0.0)  # Will softmax to uniform

        # Softmax and apply
        attn_weights = F.softmax(attn_scores, dim=-1)
        # If a row was all -inf (now all 0), softmax gives uniform 1/L_i, which is safe
        # But we want zero attention for empty segments, so mask out
        attn_weights = attn_weights.masked_fill(all_inf_mask, 0.0)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, P, d)
        attn_output = self.o_proj(attn_output)

        # Split outputs
        c_in = attn_output[:, 0, :]       # (B, d_capm)
        c_out = attn_output[:, 1, :]      # (B, d_capm)
        facets = attn_output[:, 2:, :]    # (B, K, d_capm)

        return c_in, c_out, facets

    def _build_segment_mask(self, segment_ids, P, B, L_i, device):
        """Build segment mask: CLSin sees User, CLSout sees Assistant, facets see all."""
        mask = torch.ones(B, P, L_i, device=device, dtype=torch.bool)
        mask[:, 0, :] = (segment_ids == 0)  # CLSin -> User
        mask[:, 1, :] = (segment_ids == 1)  # CLSout -> Assistant
        return mask


class FacetPooler(nn.Module):
    """Pool K facets into a single summary vector g via mean(RMSNorm(facets))."""

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_capm)

    def forward(self, facets: torch.Tensor) -> torch.Tensor:
        """(B, K, d_capm) → (B, d_capm)"""
        return self.norm(facets).mean(dim=1)


class OperatorMapper(nn.Module):
    """
    Dictionary low-rank operator conditioned on (c_in, c_out, g).

    Shared learnable bases U_base, V_base ∈ R^{d×r}.
    A small head network produces per-sample scales (u_scale, v_scale, α) ∈ R^{3r}.
    All scales are multiplied by op_gain (default 0.1) so the operator starts
    near identity: z ≈ g at initialisation.

    Transform:  z = g + U·(α ⊙ Vᵀ·g)
    where U = U_base * u_scale, V = V_base * v_scale  (element-wise broadcast).
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.config = config
        d = config.d_capm
        r = config.operator_rank
        self.op_gain = config.op_gain

        # Feature normalization
        self.feat_norm = RMSNorm(4 * d)

        # Shared bases
        self.U_base = nn.Parameter(torch.randn(d, r) * 0.02)
        self.V_base = nn.Parameter(torch.randn(d, r) * 0.02)

        # Head: (4*d) → (3*r)  produces u_scale, v_scale, alpha
        self.head_net = nn.Linear(4 * d, 3 * r)

    def forward(
        self,
        c_in: torch.Tensor,   # (B, d)
        c_out: torch.Tensor,  # (B, d)
        g: torch.Tensor,      # (B, d)
    ) -> torch.Tensor:
        """Returns z (B, d) — operator-enriched global token."""
        if getattr(self.config, "ablation_mode", "none") == "no_low_rank_transformation":
            return g

        diff = c_out - c_in
        prod = c_in * c_out
        feat_raw = torch.cat([c_in, c_out, diff, prod], dim=-1)  # (B, 4d)
        feat = self.feat_norm(feat_raw)

        scales = self.head_net(feat) * self.op_gain               # (B, 3r)
        r = scales.shape[-1] // 3
        u_scale, v_scale, alpha = scales.split(r, dim=-1)         # each (B, r)

        # Construct sample-specific operators
        U = self.U_base.unsqueeze(0) * u_scale.unsqueeze(1)       # (B, d, r)
        V = self.V_base.unsqueeze(0) * v_scale.unsqueeze(1)       # (B, d, r)

        # Apply: z = g + U·(α ⊙ Vᵀ·g)
        p = torch.einsum('bdr,bd->br', V, g)   # Vᵀg → (B, r)
        p = p * alpha                           # (B, r)
        delta = torch.einsum('bdr,br->bd', U, p)  # (B, d)
        return g + delta


class CapmEncoder(nn.Module):
    """
    Per-demo encoder: Prober → FacetPooler → OperatorMapper.

    Returns the 4 token types for bank assembly:
        z      (B, d) — operator-enriched global
        c_in   (B, d) — input CLS
        c_out  (B, d) — output CLS
        facets (B, K, d) — fine-grained facets
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.config = config
        self.prober = CapmProber(config)
        self.pooler = FacetPooler(config)
        self.operator = OperatorMapper(config)
        self.detach_operator_from_facets = config.detach_operator_from_facets

    def forward(
        self,
        demo_embeds: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z, c_in, c_out, facets)."""
        c_in, c_out, facets = self.prober(demo_embeds, segment_ids, attention_mask)
        g = self.pooler(facets)
        if self.detach_operator_from_facets:
            g = g.detach()

        if getattr(self.config, "ablation_mode", "none") == "no_decoupled_encoding":
            # Keep the bank layout unchanged while removing input/output/facet-specific
            # representations. This isolates the value of decoupled demo encoding.
            c_in = g
            c_out = g
            facets = g.unsqueeze(1).expand(-1, facets.shape[1], -1)

        z = self.operator(c_in, c_out, g)
        return z, c_in, c_out, facets


# ============================================================================
# Stage 1.5: Cross-demo interaction
# ============================================================================

class DemoInteraction(nn.Module):
    """Cross-demo self-attention on z tokens.

    After each demo is independently encoded into z_i, this module lets
    all z tokens attend to each other.  This enables:
      - Complementary enhancement: demos covering different aspects reinforce.
      - Redundancy suppression: near-duplicate demos are down-weighted.

    Architecture: pre-norm Transformer block (SA + FFN) with **zero-init
    output projections** so the module is an identity at initialisation.
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        d = config.d_capm
        self.num_heads = config.cross_attn_heads
        self.head_dim = d // self.num_heads

        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o_proj.weight)       # zero-init → identity at start

        self.norm2 = RMSNorm(d)
        self.ffn_up = nn.Linear(d, d * 4, bias=False)
        self.ffn_gate = nn.Linear(d, d * 4, bias=False)
        self.ffn_down = nn.Linear(d * 4, d, bias=False)
        nn.init.zeros_(self.ffn_down.weight)     # zero-init → identity at start

    def forward(self, z_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """z_list: list of N tensors, each (B, d). Returns list of same shape."""
        if len(z_list) <= 1:
            return z_list  # single demo, nothing to interact

        # Stack: (B, N, d)
        Z = torch.stack(z_list, dim=1)
        B, N, d = Z.shape

        # Self-attention
        residual = Z
        h = self.norm1(Z)
        Q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V)  # (B, H, N, hd)
        attn = attn.transpose(1, 2).contiguous().view(B, N, d)
        Z = residual + self.o_proj(attn)

        # SwiGLU FFN
        residual = Z
        h = self.norm2(Z)
        Z = residual + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))

        return [Z[:, i, :] for i in range(N)]


# ============================================================================
# Stage 2: Bank assembly + adaptive cosine routing
# ============================================================================

class TokenCalibrator(nn.Module):
    """
    Type-wise affine calibration for bank tokens.

    4 token types: z(0), c_in(1), c_out(2), facets(3).
    Initialised to identity: γ=1, β=0.
    """
    NUM_TYPES = 4

    def __init__(self, config: CapmConfig):
        super().__init__()
        d = config.d_capm
        self.gamma = nn.Embedding(self.NUM_TYPES, d)
        self.beta = nn.Embedding(self.NUM_TYPES, d)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(
        self,
        tokens: torch.Tensor,    # (B, S, d)
        type_ids: torch.Tensor,  # (S,) or (B, S)
    ) -> torch.Tensor:
        """B_cal = tokens * γ(type_ids) + β(type_ids)"""
        gamma = self.gamma(type_ids)  # (S, d) or (B, S, d)
        beta = self.beta(type_ids)
        return tokens * gamma + beta


class PatternAligner(nn.Module):
    """
    Dense cosine routing with **operator-conditioned temperature**.

    Steps:
        q = L2Norm(W_bridge(RMSNorm(H_in)))     (B, L, d_capm)
        k = L2Norm(B_cal)                         (B, S, d_capm)
        scores = q @ kᵀ                           (B, L, S)
        τ = 0.05 + 0.7*sigmoid(τ_logit + 0.25*tanh(τ_head(z_pool)))  ★ sigmoid smooth
        C = softmax(scores / τ) @ B_cal           (B, L, d_capm)

    Innovation: τ is no longer a global scalar but is conditioned on the
    pooled operator representation z_pool = mean(z_1, ..., z_N).  Different
    ICL paradigms require different routing sharpness:
      - Retrieval ICL (few-shot matching): sharp routing (τ~0.05-0.15)
      - Inductive ICL (pattern learning): broad routing (τ~0.3-0.6)
      - Reasoning ICL (multi-step): moderate routing (τ~0.2-0.4)
      - Compositional ICL (style+content): broad routing (τ~0.4-0.7)
    Sigmoid parameterization (τ ∈ [0.05, 0.75]) with zero-init MLP lets the
    model learn task-specific τ while avoiding hard clamp dead zones.
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.config = config
        d_p = config.d_capm
        self.norm = RMSNorm(config.d_backbone)
        self.w_bridge = nn.Linear(config.d_backbone, d_p, bias=False)
        nn.init.normal_(self.w_bridge.weight, std=0.01)

        # Temperature parameterization: sigmoid mapping to avoid hard clamp dead zones
        # tau = tau_min + (tau_max - tau_min) * sigmoid(tau_logit + delta_tau)
        # This ensures gradients always flow, even near boundaries
        self.tau_min = 0.05  # Allow sharp routing for retrieval ICL
        self.tau_max = 0.75  # Allow broad routing for inductive ICL
        # Initialize tau_logit so sigmoid(tau_logit) ≈ 0.275 → initial tau ≈ 0.16
        # sigmoid^{-1}(0.275) ≈ -1.0
        self.tau_logit = nn.Parameter(torch.tensor([-1.0]))

        # Adaptive temperature head: z_pool → Δτ
        # Zero-init to let model learn task-appropriate τ from data
        # Wide range [0.03, 1.0] supports diverse ICL paradigms
        self.tau_head = nn.Sequential(
            RMSNorm(d_p),
            nn.Linear(d_p, d_p // 4),
            nn.SiLU(),
            nn.Linear(d_p // 4, 1),
        )
        nn.init.zeros_(self.tau_head[-1].weight)
        nn.init.zeros_(self.tau_head[-1].bias)  # Neutral init, data-driven
        
        # Stats cache for FSDP-safe logging
        self._stats = {}

    def forward(
        self,
        H_in: torch.Tensor,                          # (B, L, d_backbone)
        B_cal: torch.Tensor,                          # (B, S, d_capm)
        bank_mask: Optional[torch.Tensor] = None,    # (B, S)  1=valid
        z_pool: Optional[torch.Tensor] = None,        # (B, d_capm)
    ) -> torch.Tensor:
        """Returns C (B, L, d_capm) — position-wise context vectors."""
        q = self.w_bridge(self.norm(H_in))              # (B, L, d_p)
        q = F.normalize(q, dim=-1)
        k = F.normalize(B_cal, dim=-1)                  # (B, S, d_p)

        scores = torch.matmul(q, k.transpose(-2, -1))   # (B, L, S)
        if bank_mask is not None:
            scores = scores.masked_fill(bank_mask.unsqueeze(1) == 0, -1e9)

        fixed_tau = getattr(self.config, "fixed_tau", None)

        if fixed_tau is not None:
            tau = torch.full(
                (scores.shape[0], 1, 1),
                float(fixed_tau),
                dtype=scores.dtype,
                device=scores.device,
            )
            delta_tau = None
        # Adaptive temperature with sigmoid parameterization
        elif z_pool is not None:
            delta_tau = self.tau_head(z_pool)             # (B, 1)
            delta_tau = 0.25 * torch.tanh(delta_tau)      # Δτ ∈ [-0.25, 0.25], smooth bounded
            
            # Sigmoid mapping: smooth, differentiable, no hard boundaries
            tau_sigmoid_input = self.tau_logit + delta_tau  # (B, 1)
            tau_ratio = torch.sigmoid(tau_sigmoid_input)    # (B, 1) ∈ (0, 1)
            tau = self.tau_min + (self.tau_max - self.tau_min) * tau_ratio
            tau = tau.unsqueeze(-1)                       # (B, 1, 1) for broadcast
        else:
            # Fallback: use base tau_logit without adaptation
            tau_ratio = torch.sigmoid(self.tau_logit)
            tau = self.tau_min + (self.tau_max - self.tau_min) * tau_ratio
        
        # Cache per-call tau stats for upstream aggregation (FSDP-safe)
        with torch.no_grad():
            tau_base_val = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_logit)
            self._stats['tau_base'] = tau_base_val.item()

            tau_vals = tau.reshape(-1).to(torch.float32)
            tau_count = int(tau_vals.numel())
            self._stats['tau_count'] = tau_count
            self._stats['tau_sum'] = tau_vals.sum().item()
            self._stats['tau_sq_sum'] = (tau_vals * tau_vals).sum().item()
            self._stats['tau_mean'] = tau_vals.mean().item()
            self._stats['tau_var'] = tau_vals.var(unbiased=False).item() if tau_count > 1 else 0.0
            self._stats['tau_min'] = tau_vals.min().item()
            self._stats['tau_max'] = tau_vals.max().item()
            self._stats['tau_is_fixed'] = fixed_tau is not None
            self._stats['tau_fixed'] = float(fixed_tau) if fixed_tau is not None else None

            if delta_tau is not None:
                delta_vals = delta_tau.reshape(-1).to(torch.float32)
                delta_count = int(delta_vals.numel())
                self._stats['delta_tau_count'] = delta_count
                self._stats['delta_tau_sum'] = delta_vals.sum().item()
                self._stats['delta_tau_sq_sum'] = (delta_vals * delta_vals).sum().item()
                self._stats['delta_tau_mean'] = delta_vals.mean().item()
                self._stats['delta_tau_var'] = (
                    delta_vals.var(unbiased=False).item() if delta_count > 1 else 0.0
                )
                self._stats['delta_tau_std'] = (
                    delta_vals.std(unbiased=False).item() if delta_count > 1 else 0.0
                )
                self._stats['delta_tau_min'] = delta_vals.min().item()
                self._stats['delta_tau_max'] = delta_vals.max().item()
            else:
                self._stats['delta_tau_count'] = 0
                self._stats['delta_tau_sum'] = 0.0
                self._stats['delta_tau_sq_sum'] = 0.0
                self._stats['delta_tau_mean'] = 0.0
                self._stats['delta_tau_var'] = 0.0
                self._stats['delta_tau_std'] = 0.0
                self._stats['delta_tau_min'] = 0.0
                self._stats['delta_tau_max'] = 0.0

        W = F.softmax(scores / tau, dim=-1)              # (B, L, S)
        C = torch.matmul(W, B_cal)                       # (B, L, d_p)
        return C


# ============================================================================
# Stage 3: Per-layer gate injection
# ============================================================================


class CapmGate(nn.Module):
    """
    Elementwise gating for backbone attention modulation.

    Position: after attention concat, before W_o
        SDPA → concat → Y (B,L,d_backbone) → Y' = Y ⊙ mask → W_o

    Two modes controlled by config.gate_only:
      - Full mode (gate_only=False):
            X_gate = [LN(H_in); C_tiled]  dim = d_backbone + d_capm
            mask = σ(bottleneck_MLP(X_gate))
      - Gate-only ablation (gate_only=True):
            X_gate = LN(H_in)             dim = d_backbone
            mask = σ(bottleneck_MLP(X_gate))
            C_tiled is ignored entirely — no dead parameters.

    IMPORTANT: Initialized to identity (mask ≈ 1) to preserve pretrained backbone behavior.
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.d_backbone = config.d_backbone
        self.d_capm = config.d_capm
        self.gate_only = config.gate_only

        # LayerNorm for H_in
        self.ln = nn.LayerNorm(config.d_backbone)

        # Bottleneck MLP input dimension depends on mode
        if self.gate_only:
            # H_in-only: no context concatenation
            fc1_in = config.d_backbone
        else:
            # Full: [LN(H_in); C_tiled]
            fc1_in = config.d_backbone + config.d_capm

        self.fc1 = nn.Linear(fc1_in, config.d_capm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_capm, config.d_backbone)

        # Critical: Initialize to identity (mask ≈ 1)
        # Set fc2 weight to zero and bias to positive value
        # sigmoid(2) ≈ 0.88 - more room for modulation while still preserving backbone
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.0)

    def forward(
        self,
        H_in: torch.Tensor,      # (*, d_backbone) - supports both 2D and 3D
        C_tiled: torch.Tensor    # (*, d_capm) — ignored in gate_only mode
    ) -> torch.Tensor:
        """
        Returns:
            mask: (*, d_backbone)
        """
        if self.gate_only:
            # Gate-only ablation: mask depends solely on H_in
            X_gate = self.ln(H_in)                                 # (*, d_backbone)
        else:
            # Full mode: channel concat
            X_gate = torch.cat([self.ln(H_in), C_tiled], dim=-1)  # (*, d_backbone + d_capm)

        # Bottleneck MLP
        mask = self.fc1(X_gate)
        mask = self.act(mask)
        mask = self.fc2(mask)
        mask = torch.sigmoid(mask)

        return mask

    def apply_gate(self, Y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply elementwise gate: Y' = Y ⊙ mask"""
        return Y * mask


class CAPM(nn.Module):
    """
    Context-Adaptive Prototype Modulator full pipeline.

    Usage:
        capm = CAPM(config)

        # 1) Encode demos → calibrated bank (cached)
        capm.encode_demos(demo_embeds_list, segment_ids_list)

        # 2) Route query tokens → context
        C = capm.route(query_embeds)   # (B, L_q, d_capm)

        # 3) Per-layer gating (inside decoder)
        mask = capm.compute_gate(layer_idx, H_in, C)
        Y' = capm.apply_gate(layer_idx, Y, mask)
    """

    def __init__(self, config: CapmConfig):
        super().__init__()
        self.config = config
        self.K = config.num_probes - 2  # facets per demo

        if not config.gate_only:
            # Stage 1
            self.encoder = CapmEncoder(config)

            # Stage 1.5: Cross-demo interaction
            self.demo_interaction = DemoInteraction(config)

            # Stage 2
            self.calibrator = TokenCalibrator(config)
            self.aligner = PatternAligner(config)
        else:
            # Gate-only ablation: no encoder/routing, only gates
            self.encoder = None
            self.demo_interaction = None
            self.calibrator = None
            self.aligner = None

        # Stage 3 (always instantiated)
        self.gates = nn.ModuleList([
            CapmGate(config) for _ in range(config.num_inject_layers)
        ])

        # Cache
        self._cached_bank_cal: Optional[torch.Tensor] = None
        self._cached_bank_mask: Optional[torch.Tensor] = None
        self._cached_z_pool: Optional[torch.Tensor] = None

    def encode_demos(
        self,
        demo_embeds_list: List[torch.Tensor],      # List of (B, L_i, d_backbone)
        segment_ids_list: List[torch.Tensor],
        attention_masks_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Encode N demos into calibrated pattern bank and cache.

        Each demo yields K+3 tokens: [z, c_in, c_out, facets].
        Type IDs: z=0, c_in=1, c_out=2, facets=3.

        Returns:
            B_cal: (B, S, d_capm)  where S = N × (K + 3)

        Raises:
            RuntimeError: if called in gate_only mode.
        """
        if self.config.gate_only:
            raise RuntimeError(
                "encode_demos() should not be called in gate_only mode. "
                "Gate-only ablation uses zero context; encoding is skipped."
            )
        if attention_masks_list is None:
            attention_masks_list = [None] * len(demo_embeds_list)

        all_z = []
        all_c_in = []
        all_c_out = []
        all_facets = []
        K = self.K

        for demo, seg, mask in zip(demo_embeds_list, segment_ids_list, attention_masks_list):
            z, c_in, c_out, facets = self.encoder(demo, seg, mask)
            all_z.append(z)
            all_c_in.append(c_in)
            all_c_out.append(c_out)
            all_facets.append(facets)

        # Stage 1.5: Cross-demo interaction on z tokens
        all_z = self.demo_interaction(all_z)

        # z_pool for adaptive temperature
        z_pool = torch.stack(all_z, dim=1).mean(dim=1)  # (B, d)

        # Assemble bank tokens
        all_tokens = []
        all_type_ids = []
        for z, c_in, c_out, facets in zip(all_z, all_c_in, all_c_out, all_facets):
            tokens_i = torch.cat([
                z.unsqueeze(1),        # (B, 1, d)
                c_in.unsqueeze(1),     # (B, 1, d)
                c_out.unsqueeze(1),    # (B, 1, d)
                facets,                # (B, K, d)
            ], dim=1)
            type_ids_i = torch.tensor(
                [0, 1, 2] + [3] * K,
                device=tokens_i.device, dtype=torch.long,
            )
            all_tokens.append(tokens_i)
            all_type_ids.append(type_ids_i)

        B_tokens = torch.cat(all_tokens, dim=1)    # (B, S, d)
        type_ids = torch.cat(all_type_ids, dim=0)  # (S,)

        B_cal = self.calibrator(B_tokens, type_ids)

        self._cached_bank_cal = B_cal
        self._cached_bank_mask = None  # all valid for fixed-shot
        self._cached_z_pool = z_pool   # for adaptive temperature
        return B_cal

    def route(self, query_embeds: torch.Tensor) -> torch.Tensor:
        """Dense cosine routing: query backbone tokens → context.

        Args:
            query_embeds: (B, L_q, d_backbone)
        Returns:
            C: (B, L_q, d_capm)

        Raises:
            RuntimeError: if called in gate_only mode.
        """
        if self.config.gate_only:
            raise RuntimeError(
                "route() should not be called in gate_only mode. "
                "Gate-only ablation uses zero context; routing is skipped."
            )
        if self._cached_bank_cal is None:
            raise ValueError("No bank cached. Call encode_demos first.")

        if getattr(self.config, "ablation_mode", "none") == "no_adaptive_routing":
            bank = self._cached_bank_cal
            if self._cached_bank_mask is not None:
                weights = self._cached_bank_mask.to(dtype=bank.dtype).unsqueeze(-1)
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = (bank * weights).sum(dim=1, keepdim=True) / denom
            else:
                pooled = bank.mean(dim=1, keepdim=True)
            return pooled.expand(-1, query_embeds.shape[1], -1)

        return self.aligner(
            query_embeds, self._cached_bank_cal, self._cached_bank_mask,
            z_pool=self._cached_z_pool,
        )

    def compute_gate(
        self,
        layer_idx: int,
        H_in: torch.Tensor,
        C_tiled: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gate mask for specific layer."""
        return self.gates[layer_idx](H_in, C_tiled)

    def apply_gate(
        self,
        layer_idx: int,
        Y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gate to attention output."""
        return self.gates[layer_idx].apply_gate(Y, mask)

    def clear_cache(self):
        """Clear cached bank."""
        self._cached_bank_cal = None
        self._cached_bank_mask = None
        self._cached_z_pool = None


Capm = CAPM
CAPMConfig = CapmConfig


# Backward-compatibility aliases (referenced by train/fsdp_utils.py)
CapmMapper = OperatorMapper
CapmBank = TokenCalibrator


# ============================================================================
# Utility
# ============================================================================

def count_parameters(config: CapmConfig) -> Dict[str, int]:
    """Analytical parameter count."""
    d_b = config.d_backbone
    d_p = config.d_capm
    P = config.num_probes
    r = config.operator_rank
    L = config.num_inject_layers

    prober = (
        d_b * d_p +           # input_proj
        P * d_p +             # probes
        4 * d_p * d_p         # q/k/v/o_proj
    )
    pooler = d_p              # RMSNorm weight

    operator = (
        4 * d_p +             # feat RMSNorm
        d_p * r +             # U_base
        d_p * r +             # V_base
        4 * d_p * 3 * r + 3 * r  # head_net (Linear with bias)
    )

    # DemoInteraction: SA(q/k/v/o) + FFN(up/gate/down) + 2 RMSNorm
    demo_inter = (
        4 * d_p * d_p +       # q/k/v/o_proj (no bias)
        2 * d_p * (d_p * 4) + # ffn_up + ffn_gate
        d_p * 4 * d_p +       # ffn_down
        2 * d_p               # 2x RMSNorm
    )

    calibrator = 2 * 4 * d_p  # gamma + beta embeddings (4 types)

    # PatternAligner with adaptive tau (sigmoid parameterization)
    aligner = (
        d_b +                 # RMSNorm weight
        d_b * d_p +           # w_bridge
        1 +                   # tau_logit (learnable)
        d_p +                 # tau_head RMSNorm
        d_p * (d_p // 4) + (d_p // 4) +  # tau_head fc1 + bias
        (d_p // 4) * 1 + 1              # tau_head fc2 + bias
    )

    gate = (
        d_b * 2 +                    # LayerNorm
        (d_b + d_p) * d_p + d_p +    # fc1
        d_p * d_b + d_b              # fc2
    )

    encoder_total = prober + pooler + operator
    stage2_total = calibrator + aligner + demo_inter
    gates_total = gate * L
    total = encoder_total + stage2_total + gates_total

    return {
        'prober': prober,
        'pooler': pooler,
        'operator': operator,
        'encoder': encoder_total,
        'demo_interaction': demo_inter,
        'calibrator': calibrator,
        'aligner': aligner,
        'stage2': stage2_total,
        'gate_per_layer': gate,
        'gates_total': gates_total,
        'total': total,
    }


def print_config_summary(config: CapmConfig):
    """Print configuration summary with parameter counts."""
    params = count_parameters(config)
    K = config.num_probes - 2
    T = K + 3  # tokens per demo

    print("=" * 60)
    print("CAPM Configuration Summary")
    print("=" * 60)
    print(f"\nDimensions:")
    print(f"  d_backbone:      {config.d_backbone}")
    print(f"  d_capm:         {config.d_capm}")
    print(f"  operator_rank:   {config.operator_rank}")
    print(f"  op_gain:         {config.op_gain}")
    print(f"\nArchitecture:")
    print(f"  num_probes (P):         {config.num_probes} (K={K} facets)")
    print(f"  tokens_per_demo:        {T} (1 z + 1 c_in + 1 c_out + {K} facets)")
    print(f"  cross_attn_heads:       {config.cross_attn_heads}")
    print(f"  num_inject_layers:      {config.num_inject_layers}")
    print(f"\nBank example (2-shot):")
    print(f"  S = 2 × {T} = {2 * T} bank tokens")
    print(f"\nParameters:")
    print(f"  CapmProber:        {params['prober']:>10,} ({params['prober']/1e6:.2f}M)")
    print(f"  FacetPooler:        {params['pooler']:>10,}")
    print(f"  OperatorMapper:     {params['operator']:>10,} ({params['operator']/1e6:.2f}M)")
    print(f"  Encoder total:      {params['encoder']:>10,} ({params['encoder']/1e6:.2f}M)")
    print(f"  DemoInteraction:    {params['demo_interaction']:>10,} ({params['demo_interaction']/1e6:.2f}M)")
    print(f"  TokenCalibrator:    {params['calibrator']:>10,}")
    print(f"  PatternAligner:     {params['aligner']:>10,} ({params['aligner']/1e6:.2f}M)")
    print(f"  Stage 2 total:      {params['stage2']:>10,} ({params['stage2']/1e6:.2f}M)")
    print(f"  CapmGate (×{config.num_inject_layers}):    {params['gates_total']:>10,} ({params['gates_total']/1e6:.2f}M)")
    print(f"  ─────────────────────────────────")
    print(f"  Total:              {params['total']:>10,} ({params['total']/1e6:.1f}M)")
    print("=" * 60)


if __name__ == "__main__":
    config = CapmConfig(
        d_backbone=3584,
        d_capm=768,
        num_probes=32,
        operator_rank=64,
        num_inject_layers=8,
    )
    print_config_summary(config)

    print("\nInstantiation...")
    capm = Capm(config)
    actual = sum(p.numel() for p in capm.parameters())
    print(f"Actual parameters: {actual:,} ({actual/1e6:.1f}M)")

    K = config.num_probes - 2
    print(f"\nFunctional test (2-shot, K={K})...")
    demo1 = torch.randn(1, 100, 3584)
    demo2 = torch.randn(1, 80, 3584)
    seg1 = torch.cat([torch.zeros(1, 60), torch.ones(1, 40)], dim=-1).long()
    seg2 = torch.cat([torch.zeros(1, 50), torch.ones(1, 30)], dim=-1).long()

    B_cal = capm.encode_demos([demo1, demo2], [seg1, seg2])
    S = 2 * (K + 3)
    print(f"  Bank shape: {B_cal.shape}  (expect (1, {S}, 768))")
    assert B_cal.shape == (1, S, 768), f"Bank shape mismatch: {B_cal.shape}"

    query = torch.randn(1, 200, 3584)
    C = capm.route(query)
    print(f"  Context shape: {C.shape}  (expect (1, 200, 768))")
    assert C.shape == (1, 200, 768)

    H_in = torch.randn(1, 200, 3584)
    mask = capm.compute_gate(0, H_in, C)
    Y = torch.randn(1, 200, 3584)
    Y_gated = capm.apply_gate(0, Y, mask)
    print(f"  Gate mask shape: {mask.shape}")
    print(f"  Gate mean: {mask.mean().item():.4f}")

    print(f"\n✅ All tests passed!")
