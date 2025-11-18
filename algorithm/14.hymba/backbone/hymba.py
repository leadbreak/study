"""
Hymba: Hybrid-head Architecture for Small Language Models

유연한 Mamba-Transformer 하이브리드 구현 (arXiv:2411.13676 기반)

특징:
- 레이어별 Mamba:Transformer 비율 지정 가능 (1:1, 5:1, M:N 등)
- Global/Local Attention 패턴
- Meta Tokens
- Cross-layer KV 공유
- FlexAttention (PyTorch 2.5+)

지원 구성:
- Mamba-only: 순수 SSM
- Transformer-only: 순수 Attention
- Hybrid: Mamba + Attention (비율 조정 가능)

참고:
- 논문: https://arxiv.org/abs/2411.13676
- 공식 코드: https://github.com/NVlabs/hymba
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Union

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
    warnings.warn("FlexAttention 사용 불가 (PyTorch 2.5+ 필요)")

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
    MAMBA_ONLY = "mamba"              # 순수 SSM
    TRANSFORMER_ONLY = "transformer"  # 순수 Attention
    HYBRID = "hybrid"                 # Attention + Mamba 혼합

class AttentionType(Enum):
    """어텐션 타입"""
    GLOBAL = "global"  # 전역 어텐션 (전체 시퀀스)
    LOCAL = "local"    # 로컬 어텐션 (Sliding Window)

# ===================== 기본 레이어 =====================
class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, d: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        h = int(d * mult)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))

# ===================== Rotary Position Embedding =====================
class RotaryEmbedding(nn.Module):
    """회전 위치 임베딩 (RoPE)"""
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
    """Grouped Query Attention: KV 헤드 반복"""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)

# ===================== Attention 레이어 =====================
class TransformerAttention(nn.Module):
    """Transformer Self-Attention with SWA support"""

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv: int = None,
        attn_type: AttentionType = AttentionType.GLOBAL,
        window: int = 1024,
        dropout: float = 0.0,
        num_meta_tokens: int = 0,
        rope_base: float = 10000.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d = d
        self.H = n_heads
        self.KV = n_kv if n_kv is not None else n_heads
        self.Dh = d // n_heads
        self.rep = self.H // self.KV
        self.layer_idx = layer_idx

        self.attn_type = attn_type
        self.window = window
        self.num_meta = num_meta_tokens

        self.q = nn.Linear(d, n_heads * self.Dh, bias=False)
        self.k = nn.Linear(d, self.KV * self.Dh, bias=False)
        self.v = nn.Linear(d, self.KV * self.Dh, bias=False)
        self.o = nn.Linear(n_heads * self.Dh, d, bias=False)

        self.rope = RotaryEmbedding(self.Dh, base=rope_base)
        self.drop = nn.Dropout(dropout)

        self.use_flex = HAS_FLEX_ATTN and attn_type == AttentionType.LOCAL
        if self.use_flex:
            self._setup_flex()

    def _setup_flex(self):
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
        row_idx = torch.arange(T, device=device).unsqueeze(1)
        col_idx = torch.arange(Tk, device=device).unsqueeze(0)
        offset = Tk - T
        mask = torch.where(col_idx > row_idx + offset, float('-inf'), 0.0)
        return mask

    def _make_swa_mask(self, T: int, Tk: int, device) -> torch.Tensor:
        """Sliding Window Attention 마스크"""
        mask = torch.full((T, Tk), float('-inf'), device=device)
        offset = Tk - T

        for i in range(T):
            if self.num_meta > 0:
                mask[i, :self.num_meta] = 0.0
            abs_q_pos = i + offset
            window_start = max(self.num_meta, abs_q_pos - self.window + 1)
            window_end = abs_q_pos + 1
            mask[i, window_start:window_end] = 0.0

        return mask

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.H, self.Dh).transpose(1, 2)
        k_new = self.k(x).view(B, T, self.KV, self.Dh).transpose(1, 2)
        v_new = self.v(x).view(B, T, self.KV, self.Dh).transpose(1, 2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k_new], dim=2)
            v = torch.cat([v_prev, v_new], dim=2)
        else:
            k, v = k_new, v_new

        Tc = k.size(2)

        pos_q = torch.arange(Tc - T, Tc, device=x.device)
        pos_k = torch.arange(Tc, device=x.device)
        q = self.rope.apply_rotary(q, pos_q)
        k = self.rope.apply_rotary(k, pos_k)

        new_cache = (k.detach(), v.detach())

        k_full = repeat_kv(k, self.rep)
        v_full = repeat_kv(v, self.rep)
        Tk = k_full.size(2)

        attn_weights = None

        if self.use_flex and not return_attn:
            block_mask = create_block_mask(self.flex_mask, B=None, H=None, Q_LEN=T, KV_LEN=Tk)
            out = self.flex_attn(q, k_full, v_full, block_mask=block_mask)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)

        elif HAS_FLASH_ATTN and not return_attn and self.num_meta == 0 and self.attn_type == AttentionType.GLOBAL:
            q_f = q.transpose(1, 2)
            k_f = k_full.transpose(1, 2)
            v_f = v_full.transpose(1, 2)

            input_dtype = q_f.dtype
            if input_dtype not in [torch.float16, torch.bfloat16]:
                flash_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                q_f, k_f, v_f = q_f.to(flash_dtype), k_f.to(flash_dtype), v_f.to(flash_dtype)
            else:
                flash_dtype = input_dtype

            out = flash_attn_func(q_f, k_f, v_f, causal=True)
            if flash_dtype != input_dtype:
                out = out.to(input_dtype)
            out = out.reshape(B, T, self.H * self.Dh)

        else:
            scale = 1.0 / math.sqrt(self.Dh)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale

            if self.attn_type == AttentionType.LOCAL:
                mask = self._make_swa_mask(T, Tk, x.device)
            else:
                mask = self._make_causal_mask(T, Tk, x.device)

            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn) if self.training else attn

            if return_attn:
                attn_weights = attn

            out = torch.matmul(attn, v_full)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)

        out = self.o(out)
        return out, new_cache, attn_weights

# ===================== Mamba 레이어 =====================
class MambaLayer(nn.Module):
    """Mamba State Space Model"""
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba SSM이 필요합니다: pip install mamba-ssm")
        self.mamba = MambaSSM(d_model=d, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)

# ===================== 하이브리드 블록 =====================
class HymbaBlock(nn.Module):
    """Hymba 하이브리드 블록

    유연한 Mamba:Transformer 비율 지원
    - n_mamba_heads=0: Transformer-only
    - n_attn_heads=0: Mamba-only
    - 둘 다 >0: Hybrid (비율 조정 가능)

    Fusion 공식 (Hybrid):
    Y = β_attn · norm(Attn(X)) + β_mamba · norm(Mamba(X))
    """

    def __init__(
        self,
        d_model: int,
        n_attn_heads: int = 8,
        n_mamba_heads: int = 1,
        n_kv_heads: int = 2,
        attn_type: AttentionType = AttentionType.GLOBAL,
        window: int = 1024,
        dropout: float = 0.0,
        num_meta_tokens: int = 0,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.n_attn_heads = n_attn_heads
        self.n_mamba_heads = n_mamba_heads

        # 아키텍처 타입 결정
        self.has_attn = n_attn_heads > 0
        self.has_mamba = n_mamba_heads > 0
        self.is_hybrid = self.has_attn and self.has_mamba

        self.attn_type = attn_type

        self.norm1 = RMSNorm(d_model)

        # Attention 경로
        if self.has_attn:
            self.attn = TransformerAttention(
                d_model, n_attn_heads, n_kv_heads, attn_type, window,
                dropout, num_meta_tokens, layer_idx=layer_idx
            )

        # Mamba 경로 (여러 개의 Mamba head 지원)
        if self.has_mamba:
            self.mamba_heads = nn.ModuleList([
                MambaLayer(d_model, mamba_d_state, mamba_d_conv, mamba_expand)
                for _ in range(n_mamba_heads)
            ])
            if n_mamba_heads > 1:
                # 여러 Mamba head 출력 결합
                self.mamba_proj = nn.Linear(d_model * n_mamba_heads, d_model, bias=False)

        # Hybrid fusion
        if self.is_hybrid:
            self.beta_attn = nn.Parameter(torch.ones(d_model))
            self.beta_mamba = nn.Parameter(torch.ones(d_model))
            self.attn_out_norm = RMSNorm(d_model)
            self.mamba_out_norm = RMSNorm(d_model)

        # FFN
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple] = None,
        return_attn: bool = False,
    ):
        h = self.norm1(x)
        new_cache = None
        attn_weights = None

        if self.has_attn and not self.has_mamba:
            # Transformer-only
            attn_out, new_cache, attn_w = self.attn(h, kv_cache, return_attn)
            y = attn_out
            if return_attn:
                attn_weights = attn_w

        elif self.has_mamba and not self.has_attn:
            # Mamba-only
            if self.n_mamba_heads == 1:
                y = self.mamba_heads[0](h)
            else:
                mamba_outs = [m(h) for m in self.mamba_heads]
                y = self.mamba_proj(torch.cat(mamba_outs, dim=-1))

        else:  # Hybrid
            attn_out, new_cache, attn_w = self.attn(h, kv_cache, return_attn)
            if return_attn:
                attn_weights = attn_w

            if self.n_mamba_heads == 1:
                mamba_out = self.mamba_heads[0](h)
            else:
                mamba_outs = [m(h) for m in self.mamba_heads]
                mamba_out = self.mamba_proj(torch.cat(mamba_outs, dim=-1))

            # Fusion: β_attn · norm(Attn) + β_mamba · norm(Mamba)
            attn_normalized = self.attn_out_norm(attn_out)
            mamba_normalized = self.mamba_out_norm(mamba_out)
            y = self.beta_attn * attn_normalized + self.beta_mamba * mamba_normalized

        # Residual + FFN
        x = x + self.drop(y)
        x = x + self.drop(self.ffn(self.norm2(x)))

        return x, new_cache, attn_weights

    def get_fusion_weights(self):
        """Hybrid fusion 가중치 반환"""
        if self.is_hybrid:
            return {
                "beta_attn": self.beta_attn.detach().cpu(),
                "beta_mamba": self.beta_mamba.detach().cpu(),
            }
        return None

# ===================== 모델 설정 =====================
@dataclass
class HymbaConfig:
    """Hymba 모델 설정

    유연한 Mamba:Transformer 비율 지원

    예시:
    - attn_ratio=1.0: Transformer-only
    - attn_ratio=0.0: Mamba-only
    - attn_ratio=0.5: 1:1 Hybrid
    - mamba_heads_per_layer=5, attn_ratio=0.5: 5:1 Mamba:Attn

    또는 layer_configs로 레이어별 직접 지정 가능
    """
    # 기본
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 12

    # 아키텍처 타입
    arch_type: ArchType = ArchType.HYBRID

    # Attention 설정
    n_heads: int = 8
    n_kv_heads: int = 2

    # Hybrid 비율 (0.0 = Mamba-only, 1.0 = Transformer-only)
    attn_ratio: float = 0.5

    # Mamba heads per layer (Hybrid에서 Mamba 비중 조절)
    mamba_heads_per_layer: int = 1

    # Global/Local 패턴
    global_attn_indices: Optional[List[int]] = None
    swa_window: int = 1024

    # Meta Tokens
    use_meta_tokens: bool = True
    num_meta_tokens: int = 128

    # KV 공유
    use_kv_sharing: bool = True

    # Mamba 설정
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # 기타
    dropout: float = 0.0
    seq_len: int = 512

    # 고급: 레이어별 직접 설정 (None이면 위 설정 사용)
    # 형식: [(n_attn_heads, n_mamba_heads), ...]
    layer_configs: Optional[List[Tuple[int, int]]] = None

    def get_attention_types(self) -> List[AttentionType]:
        """레이어별 어텐션 타입 계산"""
        if self.arch_type == ArchType.MAMBA_ONLY:
            return [AttentionType.GLOBAL] * self.n_layers  # placeholder

        if self.global_attn_indices is None:
            global_indices = {0, self.n_layers // 2, self.n_layers - 1}
        else:
            global_indices = set(self.global_attn_indices)

        return [
            AttentionType.GLOBAL if i in global_indices else AttentionType.LOCAL
            for i in range(self.n_layers)
        ]

    def get_layer_configs(self) -> List[Tuple[int, int]]:
        """레이어별 (n_attn_heads, n_mamba_heads) 반환"""
        if self.layer_configs is not None:
            return self.layer_configs

        # arch_type에 따라 자동 계산
        if self.arch_type == ArchType.MAMBA_ONLY:
            return [(0, self.mamba_heads_per_layer)] * self.n_layers
        elif self.arch_type == ArchType.TRANSFORMER_ONLY:
            return [(self.n_heads, 0)] * self.n_layers
        else:  # HYBRID
            return [(self.n_heads, self.mamba_heads_per_layer)] * self.n_layers

# ===================== Hymba 모델 =====================
class Hymba(nn.Module):
    """Hymba: Hybrid-head Architecture

    유연한 구성 지원:
    1. Mamba-only: 순수 SSM
    2. Transformer-only: 순수 Attention
    3. Hybrid: Mamba + Attention (비율 조정 가능)

    예시:
    - 1:1 Hybrid: HymbaConfig(arch_type=HYBRID, mamba_heads_per_layer=1)
    - 5:1 Mamba:Attn: HymbaConfig(arch_type=HYBRID, mamba_heads_per_layer=5)
    - 레이어별 커스텀: HymbaConfig(layer_configs=[(8,1), (8,2), ...])
    """

    def __init__(self, cfg: HymbaConfig):
        super().__init__()
        self.cfg = cfg

        # 메타 토큰
        self.meta_tokens = None
        if cfg.use_meta_tokens and cfg.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(1, cfg.num_meta_tokens, cfg.d_model) * 0.02)

        # 토큰 임베딩
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # 레이어별 설정
        attn_types = cfg.get_attention_types()
        layer_configs = cfg.get_layer_configs()

        # KV 공유 설정
        self.owner = list(range(cfg.n_layers))
        if cfg.use_kv_sharing and cfg.arch_type != ArchType.MAMBA_ONLY:
            i = 0
            while i < cfg.n_layers:
                if attn_types[i] == AttentionType.LOCAL:
                    j = i + 1
                    while j < cfg.n_layers and attn_types[j] == AttentionType.LOCAL:
                        self.owner[j] = i
                        j += 1
                        if j - i >= 2:
                            break
                    i = j
                else:
                    i += 1

        # 블록 생성
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            n_attn, n_mamba = layer_configs[i]

            # Mamba-only면 어텐션 설정 무시
            if n_attn == 0:
                attn_type = AttentionType.GLOBAL  # placeholder
                num_meta = 0
            else:
                attn_type = attn_types[i]
                num_meta = cfg.num_meta_tokens if cfg.use_meta_tokens else 0

            self.blocks.append(HymbaBlock(
                cfg.d_model,
                n_attn_heads=n_attn,
                n_mamba_heads=n_mamba,
                n_kv_heads=cfg.n_kv_heads if n_attn > 0 else 1,
                attn_type=attn_type,
                window=cfg.swa_window,
                dropout=cfg.dropout,
                num_meta_tokens=num_meta,
                mamba_d_state=cfg.mamba_d_state,
                mamba_d_conv=cfg.mamba_d_conv,
                mamba_expand=cfg.mamba_expand,
                layer_idx=i,
            ))

        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, return_attn=False):
        B, T = x.shape
        h = self.tok_emb(x)

        # 메타 토큰 추가
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)

        # 블록 통과
        attn_weights_list = []
        for block in self.blocks:
            h, _, attn_w = block(h, kv_cache=None, return_attn=return_attn)
            if return_attn:
                attn_weights_list.append(attn_w)

        # 메타 토큰 제거
        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:]

        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        output = {"logits": logits, "loss": loss}
        if return_attn:
            output["attn_weights"] = attn_weights_list

        return output

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """자동회귀 생성 (KV 캐시 사용)"""
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)

        B = idx.size(0)
        h = self.tok_emb(idx)

        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)

        # Prefill
        kv_caches = {}
        for i, block in enumerate(self.blocks):
            owner = self.owner[i]
            cache = kv_caches.get(owner)
            h, new_cache, _ = block(h, cache, return_attn=False)
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
                h, new_cache, _ = block(h, cache, return_attn=False)
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
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_kv_sharing_info(self):
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

    def get_attention_pattern_info(self):
        attn_types = self.cfg.get_attention_types()
        layer_configs = self.cfg.get_layer_configs()

        # 실제로 어텐션이 있는 레이어만
        attn_layers = [i for i, (n_attn, _) in enumerate(layer_configs) if n_attn > 0]

        global_layers = [i for i in attn_layers if attn_types[i] == AttentionType.GLOBAL]
        local_layers = [i for i in attn_layers if attn_types[i] == AttentionType.LOCAL]

        return {
            "total_layers": self.cfg.n_layers,
            "attn_layers": attn_layers,
            "global_layers": global_layers,
            "local_layers": local_layers,
            "num_global": len(global_layers),
            "num_local": len(local_layers),
        }

    def get_architecture_info(self):
        """아키텍처 상세 정보"""
        layer_configs = self.cfg.get_layer_configs()

        info = {
            "arch_type": self.cfg.arch_type.value,
            "n_layers": self.cfg.n_layers,
            "d_model": self.cfg.d_model,
            "layer_configs": layer_configs,
            "total_attn_heads": sum(c[0] for c in layer_configs),
            "total_mamba_heads": sum(c[1] for c in layer_configs),
        }

        if info["total_attn_heads"] > 0 and info["total_mamba_heads"] > 0:
            info["mamba_to_attn_ratio"] = info["total_mamba_heads"] / info["total_attn_heads"]

        return info

    def get_fusion_weights(self):
        """모든 레이어의 fusion 가중치 반환"""
        weights = {}
        for i, block in enumerate(self.blocks):
            w = block.get_fusion_weights()
            if w is not None:
                weights[i] = w
        return weights if weights else None
