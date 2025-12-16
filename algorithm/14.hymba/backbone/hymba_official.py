"""
Hymba Official Implementation (Simplified)

NVlabs/hymba 및 HuggingFace nvidia/Hymba-1.5B-Base 공식 구현 기반
간소화된 버전 (MoE, 캐싱 등 제외, 핵심 아키텍처만 포함)

핵심 차이점 분석을 위한 공식 구현 재현:
1. In-projection: 단일 Linear로 Q, K, V, Mamba hidden, gate 모두 projection
2. Attention: attn_only_wo_proj=True (attention 내부에 o_proj 없음)
3. Fusion: (norm(attn) + norm(mamba)) / 2
4. Out-projection: 단일 Linear로 fusion 결과 projection
5. KV reuse: kv_last_layer로 K, V 전달

참고:
- https://huggingface.co/nvidia/Hymba-1.5B-Base/blob/main/modeling_hymba.py
- arXiv:2411.13676
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    warnings.warn("mamba-ssm 패키지 없음")

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False


# ===================== RMSNorm =====================
class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ===================== Rotary Embedding =====================
class HymbaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_position_embeddings: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        seq_len = int(position_ids.max()) + 1
        if seq_len > self.cos_cached.size(0):
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, T, D]
        sin = self.sin_cached[position_ids].unsqueeze(1)  # [B, 1, T, D]
        return cos, sin


def rotate_half(x):
    """Rotary embedding helper"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to Q and K"""
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None

    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: KV 헤드 반복"""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ===================== Attention (공식 구현 스타일) =====================
class HymbaOfficialAttention(nn.Module):
    """
    공식 구현의 Attention

    특징:
    - attn_only_wo_proj=True: 외부에서 Q, K, V를 받고, o_proj 없음
    - kv_last_layer로 K, V 재사용 (reuse_kv=True일 때)
    - Q, K normalization (per-head RMSNorm)
    - RoPE 적용
    - Global/Local attention 지원 (SWA for Local)
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        reuse_kv: bool = False,
        attn_hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        is_global: bool = True,  # Global or Local attention
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.reuse_kv = reuse_kv
        self.is_global = is_global

        # Block에서 전달받은 값 또는 config 기본값 사용
        self.hidden_size = attn_hidden_size if attn_hidden_size is not None else config.attn_hidden_size
        self.num_heads = num_heads if num_heads is not None else config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attn_hidden_size // config.num_attention_heads  # 원래 head_dim 유지
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # SWA parameters for Local attention
        self.window_size = config.attn_window_size
        self.num_meta = config.num_memory_tokens

        # Q, K normalization (per-head)
        self.q_norm = HymbaRMSNorm(self.head_dim)
        self.k_norm = HymbaRMSNorm(self.head_dim)

        # RoPE
        self.rotary_emb = HymbaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_seq_len
        )

        self.attention_dropout = config.attention_dropout

    def _make_swa_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """SWA mask 생성: causal AND (in_window OR is_meta)"""
        mask = torch.full((T, T), float('-inf'), device=device)

        q_idx = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
        k_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

        # 1. Causal: k <= q
        causal = (k_idx <= q_idx)

        # 2. Window: k >= q - window + 1
        window_start = torch.clamp(q_idx - self.window_size + 1, min=0)
        in_window = (k_idx >= window_start)

        # 3. Meta: k < num_meta
        is_meta = (k_idx < self.num_meta) if self.num_meta > 0 else torch.zeros_like(k_idx, dtype=torch.bool)

        # Final: causal AND (in_window OR is_meta)
        attend = causal & (in_window | is_meta)
        mask[attend] = 0.0

        return mask

    def forward(
        self,
        query_states: torch.Tensor,  # [B, T, attn_hidden_size]
        key_states: Optional[torch.Tensor] = None,  # [B, T, k_hidden_size]
        value_states: Optional[torch.Tensor] = None,  # [B, T, v_hidden_size]
        position_ids: Optional[torch.Tensor] = None,
        kv_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Returns:
            attn_output: [B, T, attn_hidden_size]
            kv_for_next: (K, V) for next layer if not reuse_kv
            attn_weights: if return_attn_weights
        """
        B, T, _ = query_states.shape

        # Reshape Q
        q = query_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Q normalization
        q = self.q_norm(q)

        # K, V 처리
        if self.reuse_kv:
            assert kv_last_layer is not None, "reuse_kv=True but kv_last_layer is None"
            k, v = kv_last_layer
        else:
            # K, V reshape
            k = key_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = value_states.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # K normalization
            k = self.k_norm(k)

        # RoPE
        cos, sin = self.rotary_emb(q, position_ids)
        q, k_rotated = apply_rotary_pos_emb(q, k if not self.reuse_kv else None, cos, sin)

        if not self.reuse_kv:
            k = k_rotated

        # KV for next layer (before repeat_kv)
        kv_for_next = (k, v) if not self.reuse_kv else None

        # GQA: repeat KV heads
        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

        # Mask: Global uses causal, Local uses SWA
        if attention_mask is None:
            if self.is_global:
                # Simple causal mask for global attention
                mask = torch.triu(
                    torch.full((T, T), float('-inf'), device=scores.device),
                    diagonal=1
                )
            else:
                # SWA mask for local attention: causal AND (in_window OR is_meta)
                mask = self._make_swa_mask(T, scores.device)
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)

        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        attn_output = torch.matmul(attn_weights, v_expanded)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_size)

        if return_attn_weights:
            return attn_output, kv_for_next, attn_weights
        return attn_output, kv_for_next, None


# ===================== HymbaBlock (공식 구현 스타일) =====================
class HymbaOfficialBlock(nn.Module):
    """
    공식 구현의 HymbaBlock (Mamba + Attention 하이브리드)

    핵심 특징:
    1. 단일 in_proj로 모든 것을 projection (Q, K, V, Mamba hidden, gate)
    2. Mamba: Conv1d -> SSM (selective_scan)
    3. Attention: Flash/Standard
    4. Fusion: (norm(attn) + norm(mamba)) / 2
    5. 단일 out_proj로 출력
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        reuse_kv: bool = False,
        is_global: bool = True,  # Global or Local attention
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.reuse_kv = reuse_kv
        self.is_global = is_global

        self.hidden_size = config.hidden_size
        self.mamba_expand = config.mamba_expand
        self.intermediate_size = int(self.mamba_expand * self.hidden_size)

        # Attention dimensions
        # 공식 구현: attention output도 intermediate_size와 동일해야 fusion 가능
        self.attn_hidden_size = self.intermediate_size  # attention output size = mamba output size
        self.head_dim = config.attn_hidden_size // config.num_attention_heads
        self.k_hidden_size = config.num_key_value_heads * self.head_dim
        self.v_hidden_size = self.k_hidden_size

        # Mamba dimensions
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.time_step_rank = config.mamba_dt_rank

        # Calculate latent dimension
        if self.reuse_kv:
            self.latent_dim = self.intermediate_size + self.attn_hidden_size
        else:
            self.latent_dim = (
                self.intermediate_size +
                self.attn_hidden_size +
                self.k_hidden_size +
                self.v_hidden_size
            )

        # In projection: hidden_size -> latent_dim + gate_dim
        self.in_proj = nn.Linear(
            self.hidden_size,
            self.latent_dim + self.intermediate_size,  # + gate
            bias=config.mamba_proj_bias
        )

        # Attention (without o_proj)
        # attention hidden size를 intermediate_size로 맞춤
        self.num_attn_heads = self.attn_hidden_size // self.head_dim
        self.self_attn = HymbaOfficialAttention(
            config,
            layer_idx=layer_idx,
            reuse_kv=reuse_kv,
            attn_hidden_size=self.attn_hidden_size,
            num_heads=self.num_attn_heads,
            is_global=is_global,
        )

        # Mamba Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
            bias=config.mamba_conv_bias
        )

        # SSM parameters
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # A matrix (state transition)
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        # D matrix (feedthrough)
        self.D = nn.Parameter(torch.ones(self.intermediate_size))

        # Pre-average layer norms (for fusion)
        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_size)  # for attention
        self.pre_avg_layernorm2 = HymbaRMSNorm(self.intermediate_size)  # for mamba

        # Out projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mamba_proj_bias)

        # Inner layer norms (optional)
        if config.mamba_inner_layernorms:
            self.dt_layernorm = HymbaRMSNorm(self.time_step_rank)
            self.B_layernorm = HymbaRMSNorm(self.ssm_state_size)
            self.C_layernorm = HymbaRMSNorm(self.ssm_state_size)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        self.activation = "silu"

    def _apply_inner_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_last_layer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, hidden_size]
            position_ids: [B, T]
            kv_last_layer: (K, V) from producer layer (if reuse_kv)

        Returns:
            output: [B, T, hidden_size]
            kv_for_next: (K, V) for consumer layer
            attn_weights: if return_attn_weights
        """
        B, T, _ = hidden_states.shape

        # In projection
        projected = self.in_proj(hidden_states)  # [B, T, latent_dim + gate_dim]
        projected = projected.transpose(1, 2)  # [B, latent_dim + gate_dim, T]

        # Split into latent and gate
        latent, gate = projected.split([self.latent_dim, self.intermediate_size], dim=1)

        # Split latent into Q, K, V, mamba_hidden
        if self.reuse_kv:
            query_states, mamba_hidden = latent.split(
                [self.attn_hidden_size, self.intermediate_size], dim=1
            )
            query_states = query_states.transpose(1, 2)  # [B, T, attn_hidden_size]
            key_states = None
            value_states = None
        else:
            query_states, key_states, value_states, mamba_hidden = latent.split(
                [self.attn_hidden_size, self.k_hidden_size, self.v_hidden_size, self.intermediate_size],
                dim=1
            )
            query_states = query_states.transpose(1, 2)  # [B, T, attn_hidden_size]
            key_states = key_states.transpose(1, 2)  # [B, T, k_hidden_size]
            value_states = value_states.transpose(1, 2)  # [B, T, v_hidden_size]

        # ========== Mamba branch ==========
        # Conv1d
        if HAS_CAUSAL_CONV1D:
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0),
                self.conv1d.weight.size(2)
            )
            mamba_hidden = causal_conv1d_fn(
                mamba_hidden, conv_weights, self.conv1d.bias,
                activation=self.activation
            )
        else:
            # Fallback: standard conv1d
            mamba_hidden = self.conv1d(mamba_hidden)[..., :T]
            mamba_hidden = F.silu(mamba_hidden)

        # SSM parameters
        ssm_params = self.x_proj(mamba_hidden.transpose(1, 2))  # [B, T, dt_rank + 2*d_state]
        dt, B_param, C_param = ssm_params.split(
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1
        )

        # Inner layer norms
        dt, B_param, C_param = self._apply_inner_layernorms(dt, B_param, C_param)

        # dt projection
        dt_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_dt = self.dt_proj(dt).transpose(1, 2)  # [B, intermediate_size, T]
        self.dt_proj.bias = dt_bias

        # A matrix
        A = -torch.exp(self.A_log.float())

        # Selective scan
        if HAS_MAMBA:
            scan_output, _ = selective_scan_fn(
                mamba_hidden,
                discrete_dt,
                A,
                B_param.transpose(1, 2),
                C_param.transpose(1, 2),
                self.D.float(),
                z=gate,
                delta_bias=dt_bias.float() if dt_bias is not None else None,
                delta_softplus=True,
                return_last_state=True,
            )
        else:
            # Fallback: simple linear (not correct, just for testing)
            scan_output = mamba_hidden * F.sigmoid(gate)

        mamba_output = scan_output.transpose(1, 2)  # [B, T, intermediate_size]

        # ========== Attention branch ==========
        attn_output, kv_for_next, attn_weights = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            position_ids=position_ids,
            kv_last_layer=kv_last_layer,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
        )

        # ========== Fusion ==========
        # (norm(attn) + norm(mamba)) / 2
        fused = (self.pre_avg_layernorm1(attn_output) + self.pre_avg_layernorm2(mamba_output)) / 2

        # Out projection
        output = self.out_proj(fused)

        return output, kv_for_next, attn_weights


# ===================== DecoderLayer =====================
class HymbaOfficialDecoderLayer(nn.Module):
    """공식 구현의 DecoderLayer (MoE 제외)"""

    def __init__(self, config, layer_idx: int = 0, reuse_kv: bool = False, is_global: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.reuse_kv = reuse_kv
        self.is_global = is_global

        # Input layernorm
        self.input_layernorm = HymbaRMSNorm(config.hidden_size)

        # Hybrid block
        self.mamba = HymbaOfficialBlock(config, layer_idx=layer_idx, reuse_kv=reuse_kv, is_global=is_global)

        # FFN (simplified, no MoE)
        self.intermediate_size = config.intermediate_size
        if self.intermediate_size > 0:
            self.pre_moe_layernorm = HymbaRMSNorm(config.hidden_size)
            self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_last_layer: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ):
        residual = hidden_states

        # Input norm
        hidden_states = self.input_layernorm(hidden_states)

        # Hybrid block
        hidden_states, kv_for_next, attn_weights = self.mamba(
            hidden_states,
            position_ids=position_ids,
            kv_last_layer=kv_last_layer,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
        )

        # Residual
        hidden_states = residual + hidden_states

        # FFN
        if self.intermediate_size > 0:
            residual = hidden_states
            hidden_states = self.pre_moe_layernorm(hidden_states)
            hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
            hidden_states = self.down_proj(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, kv_for_next, attn_weights


# ===================== Config =====================
@dataclass
class HymbaOfficialConfig:
    """공식 Hymba 설정"""
    vocab_size: int = 32001
    hidden_size: int = 1600
    num_hidden_layers: int = 32

    # Attention
    num_attention_heads: int = 25
    num_key_value_heads: int = 5
    attn_hidden_size: Optional[int] = None  # Default: hidden_size
    attention_dropout: float = 0.0

    # Mamba
    mamba_expand: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_dt_rank: Optional[int] = None  # Default: hidden_size // 10
    mamba_proj_bias: bool = False
    mamba_conv_bias: bool = True
    mamba_inner_layernorms: bool = True

    # FFN
    intermediate_size: Optional[int] = None  # Default: mamba_expand * hidden_size

    # KV reuse
    global_attn_idx: Optional[List[int]] = None
    kv_reuse_group: Optional[List[List[int]]] = None

    # Meta tokens
    num_memory_tokens: int = 128

    # SWA (Sliding Window Attention)
    attn_window_size: int = 1024  # Local attention window size

    # Misc
    max_seq_len: int = 8192
    rms_norm_eps: float = 1e-6

    def __post_init__(self):
        # Auto-calculate dimension parameters
        if self.attn_hidden_size is None:
            self.attn_hidden_size = self.hidden_size
        if self.intermediate_size is None:
            self.intermediate_size = int(self.mamba_expand * self.hidden_size)
        if self.mamba_dt_rank is None:
            self.mamba_dt_rank = max(self.hidden_size // 10, 1)

        if self.global_attn_idx is None:
            self.global_attn_idx = [0, self.num_hidden_layers // 2, self.num_hidden_layers - 1]

        if self.kv_reuse_group is None:
            self._generate_kv_reuse_group()

    def _generate_kv_reuse_group(self):
        """KV reuse groups 자동 생성"""
        global_set = set(self.global_attn_idx)
        groups = []

        i = 0
        while i < self.num_hidden_layers:
            if i in global_set:
                i += 1
                continue

            if i + 1 < self.num_hidden_layers and (i + 1) not in global_set:
                groups.append([i, i + 1])
                i += 2
            else:
                i += 1

        self.kv_reuse_group = groups


# ===================== Model =====================
class HymbaOfficialModel(nn.Module):
    """공식 Hymba 모델 (간소화)"""

    def __init__(self, config: HymbaOfficialConfig):
        super().__init__()
        self.config = config

        # Meta tokens
        if config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(config.num_memory_tokens, config.hidden_size) * 0.02
            )
        else:
            self.memory_tokens = None

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # KV reuse map
        self.kv_reuse_map = {}  # consumer_idx -> producer_idx
        if config.kv_reuse_group:
            for group in config.kv_reuse_group:
                producer = group[0]
                for consumer in group[1:]:
                    self.kv_reuse_map[consumer] = producer

        # Decoder layers
        # Global attention layers vs Local (SWA) layers
        global_set = set(config.global_attn_idx or [])

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            reuse_kv = i in self.kv_reuse_map
            is_global = i in global_set
            self.layers.append(
                HymbaOfficialDecoderLayer(config, layer_idx=i, reuse_kv=reuse_kv, is_global=is_global)
            )

        # Final norm
        self.final_layernorm = HymbaRMSNorm(config.hidden_size)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        device = input_ids.device

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Meta tokens
        M = 0
        if self.memory_tokens is not None:
            M = self.config.num_memory_tokens
            mem = self.memory_tokens.unsqueeze(0).expand(B, -1, -1)
            hidden_states = torch.cat([mem, hidden_states], dim=1)

        # Position IDs
        position_ids = torch.arange(M + T, device=device).unsqueeze(0).expand(B, -1)

        # KV storage
        kv_store: Dict[int, Tuple] = {}
        attn_weights_list = []

        # Forward through layers
        for i, layer in enumerate(self.layers):
            # Get KV from producer if consumer
            kv_last_layer = None
            if i in self.kv_reuse_map:
                producer_idx = self.kv_reuse_map[i]
                kv_last_layer = kv_store.get(producer_idx)

            hidden_states, kv_for_next, attn_w = layer(
                hidden_states,
                position_ids=position_ids,
                kv_last_layer=kv_last_layer,
                return_attn_weights=return_attn,
            )

            # Store KV for consumers
            if kv_for_next is not None:
                kv_store[i] = kv_for_next

            if return_attn:
                attn_weights_list.append(attn_w)

        # Remove meta tokens
        if M > 0:
            hidden_states = hidden_states[:, M:]

        # Final norm and LM head
        hidden_states = self.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        result = {"logits": logits, "loss": loss}
        if return_attn:
            result["attn_weights"] = attn_weights_list

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                context = input_ids[:, -self.config.max_seq_len:]
            else:
                context = input_ids

            output = self(context)
            logits = output["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
