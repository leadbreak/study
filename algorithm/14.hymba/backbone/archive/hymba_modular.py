"""
Hymba 모듈형 구현 - Ablation Study 지원

완전히 파라미터화된 구조로 다음 구성을 지원:
1. Mamba-only: SSM 기반 시퀀스 모델링
2. Transformer-only: 표준 어텐션 기반
3. Hybrid: Attention + Mamba 혼합 (비율 조정 가능)

주요 기능:
- 메타 토큰 (on/off 가능)
- Cross-layer KV 공유 (Transformer)
- Sliding Window Attention
- FlexAttention 지원
- 동적 Hybrid 비율 조정

참고: Hymba 논문 (arXiv:2411.13676)
"""

from __future__ import annotations
import math
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== FlexAttention (PyTorch 2.5+) =====================
try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask, and_masks, or_masks
    )
    HAS_FLEX_ATTN = True
except ImportError:
    HAS_FLEX_ATTN = False

# Flash Attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# Mamba SSM
try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ===================== 아키텍처 타입 =====================
class ArchType(Enum):
    """모델 아키텍처 타입"""
    MAMBA_ONLY = "mamba"
    TRANSFORMER_ONLY = "transformer"
    HYBRID = "hybrid"

class AttentionType(Enum):
    """어텐션 타입 (Transformer/Hybrid 전용)"""
    GLOBAL = "global"  # 전역 어텐션 (전체 시퀀스)
    LOCAL = "local"    # 로컬 어텐션 (Sliding Window)

# ===================== 기본 레이어 =====================
class RMSNorm(nn.Module):
    """RMS 정규화"""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)

class SwiGLU(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, d: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        h = int(d * mult)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))

# ===================== RoPE =====================
class RotaryEmbedding(nn.Module):
    """회전 위치 임베딩"""
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("_inv", inv, persistent=False)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

    def _build(self, L: int, device, dtype):
        if self._cos is not None and self._cos.size(0) >= L:
            return
        t = torch.arange(L, device=device, dtype=self._inv.dtype)
        freqs = torch.einsum("i,j->ij", t, self._inv)
        self._cos = torch.cos(freqs).to(dtype)
        self._sin = torch.sin(freqs).to(dtype)

    def apply_rotary(self, x: torch.Tensor, pos: torch.Tensor):
        self._build(int(pos.max().item()) + 1, x.device, x.dtype)
        cos = self._cos.index_select(0, pos)[None, None, :, :]
        sin = self._sin.index_select(0, pos)[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.stack([o1, o2], dim=-1).flatten(-2)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: KV 헤드 반복"""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)

# ===================== Attention 레이어 =====================
class TransformerAttention(nn.Module):
    """
    표준 Transformer Attention
    - FlexAttention 지원
    - Sliding Window Attention
    - GQA + RoPE
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv: int = None,
        attn_type: AttentionType = AttentionType.GLOBAL,
        window: int = 256,
        dropout: float = 0.0,
        num_meta_tokens: int = 0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d = d
        self.H = n_heads
        self.KV = n_kv if n_kv is not None else n_heads
        self.Dh = d // n_heads
        self.rep = self.H // self.KV

        self.attn_type = attn_type
        self.window = window
        self.num_meta = num_meta_tokens

        # Q, K, V
        self.q = nn.Linear(d, n_heads * self.Dh, bias=False)
        self.k = nn.Linear(d, self.KV * self.Dh, bias=False)
        self.v = nn.Linear(d, self.KV * self.Dh, bias=False)
        self.o = nn.Linear(n_heads * self.Dh, d, bias=False)

        self.rope = RotaryEmbedding(self.Dh, base=rope_base)
        self.drop = nn.Dropout(dropout)

        # FlexAttention 설정
        self.use_flex = HAS_FLEX_ATTN and attn_type == AttentionType.LOCAL
        if self.use_flex:
            self._setup_flex()

    def _setup_flex(self):
        """FlexAttention 마스크 설정"""
        def causal(b, h, q, k):
            return q >= k

        def sliding(b, h, q, k):
            return q - k < self.window

        def meta(b, h, q, k):
            return k < self.num_meta

        content_mask = and_masks(causal, sliding)
        self.flex_mask = or_masks(meta, content_mask) if self.num_meta > 0 else content_mask
        self.flex_attn = torch.compile(flex_attention)

    def _make_causal_mask(self, T: int, Tk: int, device) -> torch.Tensor:
        """인과적 마스크 생성"""
        return torch.triu(torch.full((T, Tk), float('-inf'), device=device), diagonal=1)

    def _apply_window(self, k, v, T):
        """Sliding Window 적용"""
        if self.attn_type != AttentionType.LOCAL:
            return k, v

        Tk = k.size(2)
        if self.num_meta > 0:
            meta_k, meta_v = k[:, :, :self.num_meta], v[:, :, :self.num_meta]
            content_k, content_v = k[:, :, self.num_meta:], v[:, :, self.num_meta:]
            w = min(self.window, content_k.size(2))
            content_k, content_v = content_k[:, :, -w:], content_v[:, :, -w:]
            return torch.cat([meta_k, content_k], 2), torch.cat([meta_v, content_v], 2)
        else:
            w = min(self.window, Tk)
            return k[:, :, -w:], v[:, :, -w:]

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape

        # Q, K, V
        q = self.q(x).view(B, T, self.H, self.Dh).transpose(1, 2)
        k_new = self.k(x).view(B, T, self.KV, self.Dh).transpose(1, 2)
        v_new = self.v(x).view(B, T, self.KV, self.Dh).transpose(1, 2)

        # KV 캐시 병합
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k_new], dim=2)
            v = torch.cat([v_prev, v_new], dim=2)
        else:
            k, v = k_new, v_new

        Tc = k.size(2)

        # RoPE
        pos_q = torch.arange(Tc - T, Tc, device=x.device)
        pos_k = torch.arange(Tc, device=x.device)
        q = self.rope.apply_rotary(q, pos_q)
        k = self.rope.apply_rotary(k, pos_k)

        new_cache = (k.detach(), v.detach())

        # GQA
        k_full = repeat_kv(k, self.rep)
        v_full = repeat_kv(v, self.rep)

        # Sliding Window
        k_full, v_full = self._apply_window(k_full, v_full, T)
        Tk = k_full.size(2)

        # Attention
        if self.use_flex and self.training:
            block_mask = create_block_mask(self.flex_mask, B=None, H=None, Q_LEN=T, KV_LEN=Tk)
            out = self.flex_attn(q, k_full, v_full, block_mask=block_mask)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
        elif HAS_FLASH_ATTN and self.num_meta == 0:
            q_f = q.transpose(1, 2)
            k_f = k_full.transpose(1, 2)
            v_f = v_full.transpose(1, 2)
            if self.attn_type == AttentionType.LOCAL:
                out = flash_attn_func(q_f, k_f, v_f, causal=True, window_size=(self.window, self.window))
            else:
                out = flash_attn_func(q_f, k_f, v_f, causal=True)
            out = out.reshape(B, T, self.H * self.Dh)
        else:
            # Manual SDPA
            scale = 1.0 / math.sqrt(self.Dh)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
            if Tk > 1:
                scores = scores + self._make_causal_mask(T, Tk, x.device).unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn) if self.training else attn
            out = torch.matmul(attn, v_full)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)

        return self.o(out), new_cache

# ===================== Mamba 레이어 =====================
class MambaLayer(nn.Module):
    """Mamba SSM 레이어"""
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba SSM을 사용하려면 mamba-ssm 패키지를 설치하세요")
        self.mamba = MambaSSM(d_model=d, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)

# ===================== 하이브리드 블록 =====================
class HymbaBlock(nn.Module):
    """
    모듈형 Hymba 블록

    3가지 모드:
    1. Transformer-only: attn_ratio=1.0
    2. Mamba-only: attn_ratio=0.0
    3. Hybrid: 0 < attn_ratio < 1
    """

    def __init__(
        self,
        d_model: int,
        arch_type: ArchType,
        attn_ratio: float = 0.5,  # Hybrid 비율 (0=Mamba only, 1=Attn only)
        n_heads: int = 8,
        n_kv: int = 2,
        attn_type: AttentionType = AttentionType.GLOBAL,
        window: int = 256,
        dropout: float = 0.0,
        num_meta_tokens: int = 0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.arch_type = arch_type
        self.attn_ratio = attn_ratio

        self.norm1 = RMSNorm(d_model)

        # Attention 경로
        self.has_attn = arch_type in [ArchType.TRANSFORMER_ONLY, ArchType.HYBRID]
        if self.has_attn:
            attn_dim = d_model if arch_type == ArchType.TRANSFORMER_ONLY else int(d_model * attn_ratio)
            self.to_attn = nn.Linear(d_model, attn_dim, bias=False)
            self.attn = TransformerAttention(
                attn_dim, n_heads, n_kv, attn_type, window, dropout, num_meta_tokens
            )
            self.proj_attn = nn.Linear(attn_dim, d_model, bias=False)

        # Mamba 경로
        self.has_mamba = arch_type in [ArchType.MAMBA_ONLY, ArchType.HYBRID]
        if self.has_mamba:
            mamba_dim = d_model if arch_type == ArchType.MAMBA_ONLY else int(d_model * (1 - attn_ratio))
            self.to_mamba = nn.Linear(d_model, mamba_dim, bias=False)
            self.mamba = MambaLayer(mamba_dim, mamba_d_state, mamba_d_conv, mamba_expand)
            self.proj_mamba = nn.Linear(mamba_dim, d_model, bias=False)

        # Hybrid fusion
        if arch_type == ArchType.HYBRID:
            self.gate = nn.Parameter(torch.tensor(attn_ratio))

        # FFN
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        h = self.norm1(x)
        new_cache = None

        if self.arch_type == ArchType.TRANSFORMER_ONLY:
            # Transformer only
            attn_in = self.to_attn(h)
            attn_out, new_cache = self.attn(attn_in, kv_cache)
            y = self.proj_attn(attn_out)

        elif self.arch_type == ArchType.MAMBA_ONLY:
            # Mamba only
            mamba_in = self.to_mamba(h)
            mamba_out = self.mamba(mamba_in)
            y = self.proj_mamba(mamba_out)

        else:  # HYBRID
            attn_in = self.to_attn(h)
            attn_out, new_cache = self.attn(attn_in, kv_cache)
            attn_out = self.proj_attn(attn_out)

            mamba_in = self.to_mamba(h)
            mamba_out = self.mamba(mamba_in)
            mamba_out = self.proj_mamba(mamba_out)

            # Learned gate fusion
            g = torch.sigmoid(self.gate)
            y = g * attn_out + (1 - g) * mamba_out

        # Residual
        x = x + self.drop(y)
        x = x + self.drop(self.ffn(self.norm2(x)))

        return x, new_cache

# ===================== 모델 설정 =====================
@dataclass
class HymbaConfig:
    """모듈형 Hymba 설정"""
    # 기본
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 12

    # 아키텍처 타입
    arch_type: ArchType = ArchType.HYBRID

    # Transformer 설정
    n_heads: int = 8
    n_kv_heads: int = 2

    # 레이어별 어텐션 타입 (None=모두 같음, List=레이어별 지정)
    layer_attn_types: Optional[List[AttentionType]] = None

    # Hybrid 비율 (0.0=Mamba only, 1.0=Attn only)
    attn_ratio: float = 0.5

    # 메타 토큰
    use_meta_tokens: bool = True
    num_meta_tokens: int = 128

    # Sliding Window
    swa_window: int = 256

    # Cross-layer KV 공유 (Transformer/Hybrid만)
    use_kv_sharing: bool = True

    # Mamba 설정
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # 기타
    dropout: float = 0.0
    seq_len: int = 512

# ===================== Hymba 모델 =====================
class HymbaModel(nn.Module):
    """
    모듈형 Hymba 모델

    지원 구성:
    1. Mamba-only: arch_type=MAMBA_ONLY
    2. Transformer-only: arch_type=TRANSFORMER_ONLY
    3. Hybrid: arch_type=HYBRID, attn_ratio 조정
    """

    def __init__(self, cfg: HymbaConfig):
        super().__init__()
        self.cfg = cfg

        # 메타 토큰
        self.meta_tokens = None
        if cfg.use_meta_tokens and cfg.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(1, cfg.num_meta_tokens, cfg.d_model))

        # 토큰 임베딩
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # 레이어별 어텐션 타입 결정
        if cfg.layer_attn_types is None:
            # 기본: 논문 패턴 (첫/중간/마지막=Global, 나머지=Local)
            mid = cfg.n_layers // 2
            layer_attn_types = [
                AttentionType.GLOBAL if i in [0, mid, cfg.n_layers - 1] else AttentionType.LOCAL
                for i in range(cfg.n_layers)
            ]
        else:
            layer_attn_types = cfg.layer_attn_types

        # KV 공유 설정
        self.owner = list(range(cfg.n_layers))
        if cfg.use_kv_sharing and cfg.arch_type != ArchType.MAMBA_ONLY:
            # Local 레이어끼리 페어링
            i = 0
            while i < cfg.n_layers:
                if layer_attn_types[i] == AttentionType.LOCAL:
                    # 연속된 Local 레이어 찾기
                    j = i + 1
                    while j < cfg.n_layers and layer_attn_types[j] == AttentionType.LOCAL:
                        self.owner[j] = i  # j가 i의 캐시 사용
                        j += 1
                        if j - i >= 2:  # 최대 2개씩 페어
                            break
                    i = j
                else:
                    i += 1

        # 블록 생성
        self.blocks = nn.ModuleList([
            HymbaBlock(
                cfg.d_model,
                cfg.arch_type,
                cfg.attn_ratio,
                cfg.n_heads,
                cfg.n_kv_heads,
                layer_attn_types[i],
                cfg.swa_window,
                cfg.dropout,
                cfg.num_meta_tokens if cfg.use_meta_tokens else 0,
                cfg.mamba_d_state,
                cfg.mamba_d_conv,
                cfg.mamba_expand,
            )
            for i in range(cfg.n_layers)
        ])

        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.tok_emb(x)

        # 메타 토큰 추가
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)

        # 블록 통과
        for block in self.blocks:
            h, _ = block(h, kv_cache=None)

        # 메타 토큰 제거
        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:]

        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """KV 캐시 사용 생성"""
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)

        B = idx.size(0)
        h = self.tok_emb(idx)

        # 메타 토큰
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)

        # Prefill
        kv_caches = {}
        for i, block in enumerate(self.blocks):
            owner = self.owner[i]
            cache = kv_caches.get(owner)
            h, new_cache = block(h, cache)
            if i == owner and new_cache is not None:
                kv_caches[owner] = new_cache

        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:]

        logits = self.head(self.norm(h))[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        next_token = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, next_token], 1)

        # 생성 루프
        for _ in range(max_new_tokens - 1):
            h = self.tok_emb(next_token)
            for i, block in enumerate(self.blocks):
                owner = self.owner[i]
                cache = kv_caches.get(owner)
                h, new_cache = block(h, cache)
                if i == owner and new_cache is not None:
                    kv_caches[owner] = new_cache

            logits = self.head(self.norm(h))[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            next_token = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, next_token], 1)

        return idx

    def count_parameters(self):
        """파라미터 수 계산"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_kv_sharing_info(self):
        """KV 공유 정보"""
        groups = {}
        for i, owner in enumerate(self.owner):
            if owner not in groups:
                groups[owner] = []
            groups[owner].append(i)

        total_layers = len(self.owner)
        independent_caches = len([g for g in groups.values() if len(g) > 0])
        reduction = total_layers / independent_caches if independent_caches > 0 else 1.0

        return {
            "total_layers": total_layers,
            "independent_caches": independent_caches,
            "reduction": reduction,
            "groups": groups,
        }
