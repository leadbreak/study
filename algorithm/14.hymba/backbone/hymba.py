"""
Hymba: Hybrid-head Architecture for Small Language Models

공식 논문 (arXiv:2411.13676) 및 NVlabs/hymba 구현 기반 정확한 재구현

핵심 아키텍처:
1. Hybrid-head Parallel Block: Attention + Mamba 병렬 결합
2. Cross-layer KV Sharing: 연속 2개 local 레이어가 KV 공유 (producer-consumer)
3. Global/Local Attention: 첫/중간/마지막 레이어만 Global, 나머지 SWA
4. Meta Tokens: Attention sink 해결을 위한 학습 가능 토큰 (128개)

공식 구현 기반 핵심 사항:
- KV reuse: Consumer 레이어는 Q만 계산, K/V는 producer에서 재사용
- Fusion: (norm(attn) + norm(mamba)) / 2 (단순 평균)
- Local 레이어는 2의 배수여야 함 (KV 공유 쌍 형성)
- num_mamba=1 (Mamba는 레이어당 1개)

참고:
- 논문: https://arxiv.org/abs/2411.13676
- 공식 코드: https://github.com/NVlabs/hymba
- HuggingFace: https://huggingface.co/nvidia/Hymba-1.5B-Base
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 외부 라이브러리 임포트 =====================
try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask, and_masks, or_masks
    )
    HAS_FLEX_ATTN = True
except ImportError:
    HAS_FLEX_ATTN = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    warnings.warn("Mamba SSM 사용 불가 (mamba-ssm 패키지 필요)")


# ===================== 아키텍처 타입 정의 =====================
class ArchType(Enum):
    """모델 아키텍처 타입"""
    MAMBA_ONLY = "mamba"
    TRANSFORMER_ONLY = "transformer"
    HYBRID = "hybrid"


class AttentionType(Enum):
    """어텐션 타입"""
    GLOBAL = "global"
    LOCAL = "local"


# ===================== 기본 레이어 =====================
class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x * rms).to(dtype)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, d: int, mult: float = 8/3, dropout: float = 0.0):
        super().__init__()
        h = int(d * mult)
        # 8/3배로 intermediate_size 계산 (공식 구현: 5504 for d=1600)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


# ===================== Rotary Position Embedding =====================
class RotaryEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding)"""
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to x"""
        seq_len = int(position_ids.max().item()) + 1
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)

        cos = self.cos_cache[position_ids].unsqueeze(1)  # [B, 1, T, D/2]
        sin = self.sin_cache[position_ids].unsqueeze(1)  # [B, 1, T, D/2]

        # x: [B, H, T, D]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: KV 헤드 반복"""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


# ===================== Attention 레이어 =====================
class HymbaAttention(nn.Module):
    """
    Hymba Attention with KV Reuse Support

    공식 구현 기반:
    - reuse_kv=True: Q만 projection (consumer 레이어)
    - reuse_kv=False: Q, K, V 모두 projection (producer/global 레이어)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        attn_type: AttentionType = AttentionType.GLOBAL,
        window: int = 1024,
        num_meta_tokens: int = 0,
        reuse_kv: bool = False,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.attn_type = attn_type
        self.window = window
        self.num_meta = num_meta_tokens
        self.reuse_kv = reuse_kv
        self.layer_idx = layer_idx

        # Q projection은 항상 필요
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # KV projection은 reuse_kv=False일 때만 (producer 레이어)
        if not reuse_kv:
            self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Per-head RMSNorm (공식 구현)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.dropout = nn.Dropout(dropout)

        # FlexAttention setup (GQA에서는 복잡해지므로 standard attention 사용)
        head_dim_power_of_2 = (self.head_dim & (self.head_dim - 1)) == 0
        is_gqa = n_heads != n_kv_heads
        self.use_flex = HAS_FLEX_ATTN and attn_type == AttentionType.LOCAL and head_dim_power_of_2 and not is_gqa
        if self.use_flex:
            self._setup_flex_mask()

    def _setup_flex_mask(self):
        """FlexAttention 마스크 설정"""
        window = self.window
        num_meta = self.num_meta

        def causal_mask(b, h, q, k):
            return q >= k

        def window_mask(b, h, q, k):
            return q - k < window

        def meta_mask(b, h, q, k):
            return k < num_meta

        content_mask = and_masks(causal_mask, window_mask)
        self.flex_mask_fn = or_masks(meta_mask, content_mask) if num_meta > 0 else content_mask
        self.flex_attn = torch.compile(flex_attention)

    def _make_swa_mask(self, T: int, Tk: int, device: torch.device) -> torch.Tensor:
        """Sliding Window Attention 마스크 생성 (벡터화)"""
        mask = torch.full((T, Tk), float('-inf'), device=device)
        offset = Tk - T

        # Meta tokens: 항상 attend 가능
        if self.num_meta > 0:
            mask[:, :self.num_meta] = 0.0

        # Window mask (vectorized)
        q_idx = torch.arange(T, device=device).unsqueeze(1)
        k_idx = torch.arange(Tk, device=device).unsqueeze(0)
        abs_q_pos = q_idx + offset

        window_start = torch.clamp(abs_q_pos - self.window + 1, min=self.num_meta)
        in_window = (k_idx >= window_start) & (k_idx <= abs_q_pos)
        mask[in_window] = 0.0

        return mask

    def _make_causal_mask(self, T: int, Tk: int, device: torch.device) -> torch.Tensor:
        """Causal mask 생성"""
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(Tk, device=device).unsqueeze(0)
        offset = Tk - T
        return torch.where(col_idx > row_idx + offset, float('-inf'), 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, D]
            position_ids: [B, T]
            kv_cache: 이전 KV cache (incremental decoding)
            shared_kv: Producer 레이어에서 공유받은 KV (reuse_kv=True일 때)
            return_attn: attention weights 반환 여부
        """
        B, T, _ = hidden_states.shape

        # Q projection
        q = self.q_proj(hidden_states)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        # K, V 처리
        if self.reuse_kv:
            # Consumer 레이어: shared_kv 사용
            if shared_kv is None:
                raise ValueError(f"Layer {self.layer_idx}: reuse_kv=True but shared_kv is None")
            k, v = shared_kv
        else:
            # Producer/Global 레이어: K, V 직접 계산
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)

        # KV cache 처리
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv_cache = (k.detach(), v.detach()) if not self.reuse_kv else None

        # RoPE 적용 전 KV 저장 (producer가 consumer에게 전달할 원본)
        # KV sharing은 RoPE 적용 전의 K, V를 공유
        k_for_sharing = k if not self.reuse_kv else None
        v_for_sharing = v if not self.reuse_kv else None

        # RoPE 적용
        Tk = k.size(2)
        pos_q = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0).expand(B, -1)
        pos_k = torch.arange(Tk, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        q = self.rope(q, pos_q[:, -T:])
        k_rotated = self.rope(k, pos_k)

        # GQA: KV 헤드 확장
        k_expanded = repeat_kv(k_rotated, self.n_rep)
        v_expanded = repeat_kv(v, self.n_rep)

        attn_weights = None

        # Attention 계산
        if self.use_flex and not return_attn:
            block_mask = create_block_mask(self.flex_mask_fn, B=None, H=None, Q_LEN=T, KV_LEN=Tk)
            out = self.flex_attn(q, k_expanded, v_expanded, block_mask=block_mask)
        elif HAS_FLASH_ATTN and not return_attn and self.attn_type == AttentionType.GLOBAL:
            # Flash Attention (Global only)
            q_flash = q.transpose(1, 2)
            k_flash = k_expanded.transpose(1, 2)
            v_flash = v_expanded.transpose(1, 2)

            orig_dtype = q_flash.dtype
            if orig_dtype not in [torch.float16, torch.bfloat16]:
                flash_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                q_flash, k_flash, v_flash = q_flash.to(flash_dtype), k_flash.to(flash_dtype), v_flash.to(flash_dtype)

            out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)
            out = out.to(orig_dtype).transpose(1, 2)
        else:
            # Standard attention
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale

            if self.attn_type == AttentionType.LOCAL:
                mask = self._make_swa_mask(T, Tk, hidden_states.device)
            else:
                mask = self._make_causal_mask(T, Tk, hidden_states.device)

            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs) if self.training else attn_probs

            if return_attn:
                attn_weights = attn_probs

            out = torch.matmul(attn_probs, v_expanded)

        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(out)

        # Producer가 Consumer에게 전달할 KV (RoPE 적용 전, repeat_kv 전)
        produced_kv = (k_for_sharing, v_for_sharing) if k_for_sharing is not None else None

        return out, new_kv_cache, attn_weights, produced_kv


# ===================== Mamba 레이어 =====================
class MambaBlock(nn.Module):
    """Mamba SSM Block"""
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_inner_layernorms: bool = True,
    ):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba-ssm 패키지가 필요합니다: pip install mamba-ssm")

        self.mamba = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Inner layernorms (공식 구현: mamba_inner_layernorms=true)
        self.use_inner_layernorms = use_inner_layernorms
        if use_inner_layernorms:
            self.dt_layernorm = RMSNorm(d_model * expand)
            self.b_layernorm = RMSNorm(d_state)
            self.c_layernorm = RMSNorm(d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


# ===================== Hymba 하이브리드 블록 =====================
class HymbaBlock(nn.Module):
    """
    Hymba Hybrid Block: Attention + Mamba 병렬 결합

    공식 구현 기반 Fusion:
    output = (norm(attn_out) + norm(mamba_out)) / 2

    레이어 타입:
    - HYBRID: Attention + Mamba
    - ATTENTION_ONLY: Attention만
    - MAMBA_ONLY: Mamba만
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        attn_type: AttentionType,
        window: int,
        num_meta_tokens: int,
        reuse_kv: bool,
        has_attention: bool = True,
        has_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.has_attention = has_attention
        self.has_mamba = has_mamba
        self.is_hybrid = has_attention and has_mamba
        self.reuse_kv = reuse_kv
        self.attn_type = attn_type

        # Input normalization
        self.input_layernorm = RMSNorm(d_model)

        # Attention path
        if has_attention:
            self.attention = HymbaAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                attn_type=attn_type,
                window=window,
                num_meta_tokens=num_meta_tokens,
                reuse_kv=reuse_kv,
                dropout=dropout,
                layer_idx=layer_idx,
            )

        # Mamba path
        if has_mamba:
            self.mamba = MambaBlock(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )

        # Fusion normalization (공식 구현)
        if self.is_hybrid:
            self.attn_out_norm = RMSNorm(d_model)
            self.mamba_out_norm = RMSNorm(d_model)

        # FFN
        self.post_attention_layernorm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor], Optional[Tuple]]:
        """
        Returns:
            hidden_states: 출력
            new_kv_cache: 새 KV cache (producer만)
            attn_weights: attention weights
            produced_kv: Consumer에게 공유할 KV (producer만)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        new_kv_cache = None
        attn_weights = None
        produced_kv = None

        if self.has_attention and not self.has_mamba:
            # Attention-only
            attn_out, new_kv_cache, attn_weights, produced_kv = self.attention(
                hidden_states, position_ids, kv_cache, shared_kv, return_attn
            )
            hidden_states = attn_out

        elif self.has_mamba and not self.has_attention:
            # Mamba-only
            hidden_states = self.mamba(hidden_states)

        else:
            # Hybrid: 병렬 처리 후 평균
            attn_out, new_kv_cache, attn_weights, produced_kv = self.attention(
                hidden_states, position_ids, kv_cache, shared_kv, return_attn
            )
            mamba_out = self.mamba(hidden_states)

            # Fusion: (norm(attn) + norm(mamba)) / 2 (공식 구현)
            hidden_states = (self.attn_out_norm(attn_out) + self.mamba_out_norm(mamba_out)) / 2

        # Residual + FFN
        hidden_states = residual + self.dropout(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states, new_kv_cache, attn_weights, produced_kv


# ===================== 모델 설정 =====================
@dataclass
class HymbaConfig:
    """
    Hymba 모델 설정

    공식 설정 (Hymba-1.5B-Base) 기준:
    - num_hidden_layers: 32
    - hidden_size: 1600
    - num_attention_heads: 25
    - num_key_value_heads: 5
    - global_attn_idx: [0, 15, 31]
    - kv_reuse_group: [[1,2], [3,4], ..., [29,30]]
    - num_memory_tokens: 128
    - sliding_window: 1024
    - num_mamba: 1
    """
    # 기본
    vocab_size: int = 32001
    d_model: int = 1600
    n_layers: int = 32

    # Attention 설정
    n_heads: int = 25
    n_kv_heads: int = 5

    # 아키텍처 타입
    arch_type: ArchType = ArchType.HYBRID

    # Global/Local 패턴
    # Global attention 적용 레이어 인덱스 (첫/중간/마지막)
    global_attn_idx: Optional[List[int]] = None
    swa_window: int = 1024

    # Meta Tokens
    use_meta_tokens: bool = True
    num_meta_tokens: int = 128

    # KV Sharing
    # Producer-Consumer 쌍 리스트: [[producer, consumer], ...]
    # None이면 자동 생성 (연속 local 레이어 2개씩)
    kv_reuse_groups: Optional[List[List[int]]] = None

    # Mamba 설정
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # 기타
    dropout: float = 0.0
    max_seq_len: int = 8192

    def __post_init__(self):
        """설정 검증 및 자동 생성"""
        # Global attention 인덱스 자동 설정
        if self.global_attn_idx is None:
            self.global_attn_idx = [0, self.n_layers // 2, self.n_layers - 1]

        # KV reuse groups 자동 생성
        if self.kv_reuse_groups is None and self.arch_type != ArchType.MAMBA_ONLY:
            self._generate_kv_reuse_groups()

        # 검증
        self._validate()

    def _generate_kv_reuse_groups(self):
        """
        KV reuse groups 자동 생성

        공식 Hymba 구현 기반 규칙:
        - Global 레이어는 KV 공유하지 않음 (독립 KV)
        - 연속된 Local 레이어들을 2개씩 그룹으로 묶음
        - 홀수개의 연속 Local 레이어가 있으면 마지막 그룹은 3개
        - 그룹의 첫 번째가 Producer, 나머지가 Consumer

        예시 (32레이어, global=[0,15,31]):
        - Layer 1-14: [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14]]
        - Layer 16-30: [[16,17,18], [19,20], ...] (15개를 7+8로 나눔)
        """
        global_set = set(self.global_attn_idx)
        groups = []

        # 연속된 Local 레이어 구간 찾기
        i = 0
        while i < self.n_layers:
            if i in global_set:
                i += 1
                continue

            # 연속된 Local 레이어 수집
            local_segment = []
            while i < self.n_layers and i not in global_set:
                local_segment.append(i)
                i += 1

            # 이 구간을 2개씩 그룹으로 묶기
            # 홀수개면 첫 번째 그룹을 3개로 (공식 구현 방식)
            n_local = len(local_segment)
            if n_local == 0:
                continue
            elif n_local == 1:
                # 단독 레이어 - KV 공유 없음 (Producer로 취급, 그룹에 미포함)
                pass
            elif n_local % 2 == 1:
                # 홀수: 첫 그룹을 3개로
                groups.append([local_segment[0], local_segment[1], local_segment[2]])
                for k in range(3, n_local, 2):
                    groups.append([local_segment[k], local_segment[k+1]])
            else:
                # 짝수: 모두 2개씩
                for k in range(0, n_local, 2):
                    groups.append([local_segment[k], local_segment[k+1]])

        self.kv_reuse_groups = groups

    def _validate(self):
        """설정 검증"""
        if self.arch_type == ArchType.TRANSFORMER_ONLY:
            return

        if self.arch_type == ArchType.MAMBA_ONLY:
            return

        # 모든 Local 레이어가 그룹에 포함되었는지 확인
        global_set = set(self.global_attn_idx)
        local_layers = set(i for i in range(self.n_layers) if i not in global_set)
        grouped_layers = set()
        for group in (self.kv_reuse_groups or []):
            grouped_layers.update(group)

        ungrouped = local_layers - grouped_layers
        if ungrouped:
            warnings.warn(
                f"일부 Local 레이어가 KV reuse 그룹에 포함되지 않음: {sorted(ungrouped)}. "
                f"이 레이어들은 독립 KV를 사용합니다."
            )

    def get_attention_types(self) -> List[AttentionType]:
        """레이어별 attention 타입 반환"""
        global_set = set(self.global_attn_idx or [])
        return [
            AttentionType.GLOBAL if i in global_set else AttentionType.LOCAL
            for i in range(self.n_layers)
        ]

    def get_kv_reuse_map(self) -> Dict[int, int]:
        """
        Consumer → Producer 매핑 반환

        Returns:
            {consumer_idx: producer_idx, ...}
        """
        if not self.kv_reuse_groups:
            return {}

        reuse_map = {}
        for group in self.kv_reuse_groups:
            producer = group[0]
            for consumer in group[1:]:
                reuse_map[consumer] = producer
        return reuse_map

    def is_kv_producer(self, layer_idx: int) -> bool:
        """해당 레이어가 KV producer인지 확인"""
        if not self.kv_reuse_groups:
            return True

        for group in self.kv_reuse_groups:
            if layer_idx == group[0]:
                return True

        # Global 레이어는 항상 producer
        return layer_idx in (self.global_attn_idx or [])

    def is_kv_consumer(self, layer_idx: int) -> bool:
        """해당 레이어가 KV consumer인지 확인"""
        reuse_map = self.get_kv_reuse_map()
        return layer_idx in reuse_map


# ===================== Hymba 모델 =====================
class Hymba(nn.Module):
    """
    Hymba: Hybrid-head Architecture for Small Language Models

    공식 구현(NVlabs/hymba) 기반 정확한 재구현
    """

    def __init__(self, config: HymbaConfig):
        super().__init__()
        self.config = config

        # Meta tokens
        self.meta_tokens = None
        if config.use_meta_tokens and config.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(
                torch.randn(1, config.num_meta_tokens, config.d_model) * 0.02
            )

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # 레이어별 설정
        attn_types = config.get_attention_types()
        kv_reuse_map = config.get_kv_reuse_map()

        # 블록 생성
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            has_attention = config.arch_type != ArchType.MAMBA_ONLY
            has_mamba = config.arch_type != ArchType.TRANSFORMER_ONLY
            reuse_kv = i in kv_reuse_map

            self.layers.append(HymbaBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                attn_type=attn_types[i],
                window=config.swa_window,
                num_meta_tokens=config.num_meta_tokens if config.use_meta_tokens else 0,
                reuse_kv=reuse_kv,
                has_attention=has_attention,
                has_mamba=has_mamba,
                mamba_d_state=config.mamba_d_state,
                mamba_d_conv=config.mamba_d_conv,
                mamba_expand=config.mamba_expand,
                dropout=config.dropout,
                layer_idx=i,
            ))

        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # KV reuse 정보 저장
        self.kv_reuse_map = kv_reuse_map
        self.kv_producer_to_consumers = self._build_producer_consumer_map()

        self.apply(self._init_weights)

    def _build_producer_consumer_map(self) -> Dict[int, List[int]]:
        """Producer → Consumer 리스트 매핑"""
        producer_map = {}
        for consumer, producer in self.kv_reuse_map.items():
            if producer not in producer_map:
                producer_map[producer] = []
            producer_map[producer].append(consumer)
        return producer_map

    def _init_weights(self, module: nn.Module):
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
        """
        Forward pass

        Args:
            input_ids: [B, T] 입력 토큰 ID
            targets: [B, T] 타겟 토큰 ID (학습 시)
            return_attn: attention weights 반환 여부
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # Meta tokens 추가
        M = 0
        if self.meta_tokens is not None:
            M = self.config.num_meta_tokens
            meta = self.meta_tokens.expand(B, -1, -1)
            hidden_states = torch.cat([meta, hidden_states], dim=1)

        # Position IDs
        position_ids = torch.arange(M + T, device=device).unsqueeze(0).expand(B, -1)

        # KV 공유 저장소
        shared_kv_store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        attn_weights_list = []

        # Layer forward
        for i, layer in enumerate(self.layers):
            # Consumer 레이어면 producer의 KV 가져오기
            shared_kv = None
            if i in self.kv_reuse_map:
                producer_idx = self.kv_reuse_map[i]
                shared_kv = shared_kv_store.get(producer_idx)

            hidden_states, new_kv, attn_w, produced_kv = layer(
                hidden_states,
                position_ids,
                kv_cache=None,
                shared_kv=shared_kv,
                return_attn=return_attn,
            )

            # Producer 레이어면 KV 저장
            if produced_kv is not None and i in self.kv_producer_to_consumers:
                shared_kv_store[i] = produced_kv

            if return_attn:
                attn_weights_list.append(attn_w)

        # Meta tokens 제거
        if M > 0:
            hidden_states = hidden_states[:, M:]

        # Output
        hidden_states = self.norm(hidden_states)
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
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """자동회귀 생성"""
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)

        # 전체 시퀀스에 대해 forward
        for _ in range(max_new_tokens):
            # 최대 시퀀스 길이 제한
            if input_ids.size(1) > self.config.max_seq_len:
                context = input_ids[:, -self.config.max_seq_len:]
            else:
                context = input_ids

            output = self(context)
            logits = output["logits"][:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> Dict[str, int]:
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_kv_sharing_info(self) -> Dict[str, Any]:
        """KV sharing 정보 반환"""
        config = self.config

        # Producer 수 계산
        global_layers = set(config.global_attn_idx or [])
        producer_layers = set()

        for group in (config.kv_reuse_groups or []):
            producer_layers.add(group[0])
        producer_layers.update(global_layers)

        # Consumer 수
        consumer_layers = set(self.kv_reuse_map.keys())

        total_attn_layers = sum(
            1 for layer in self.layers if layer.has_attention
        )

        return {
            "total_layers": config.n_layers,
            "attention_layers": total_attn_layers,
            "producer_layers": sorted(producer_layers),
            "consumer_layers": sorted(consumer_layers),
            "num_independent_kv": len(producer_layers),
            "reduction": total_attn_layers / len(producer_layers) if producer_layers else 1.0,
            "kv_reuse_groups": config.kv_reuse_groups,
        }

    def get_attention_pattern_info(self) -> Dict[str, Any]:
        """Attention 패턴 정보 반환"""
        config = self.config
        attn_types = config.get_attention_types()

        global_layers = [i for i, t in enumerate(attn_types) if t == AttentionType.GLOBAL]
        local_layers = [i for i, t in enumerate(attn_types) if t == AttentionType.LOCAL]

        return {
            "global_layers": global_layers,
            "local_layers": local_layers,
            "num_global": len(global_layers),
            "num_local": len(local_layers),
            "window_size": config.swa_window,
        }

    def get_architecture_info(self) -> Dict[str, Any]:
        """아키텍처 정보 반환"""
        config = self.config

        return {
            "arch_type": config.arch_type.value,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "num_meta_tokens": config.num_meta_tokens if config.use_meta_tokens else 0,
            "swa_window": config.swa_window,
            "mamba_d_state": config.mamba_d_state,
        }