"""
Hymba v4: 공식 NVIDIA 구현 기반의 하이브리드 아키텍처

주요 개선사항:
1. FlexAttention 완전 통합 (PyTorch 2.5+)
2. 128개 메타 토큰 지원 (공식 구현과 동일)
3. Cross-layer KV 공유 최적화
4. Sliding Window Attention (SWA) 개선
5. Mamba SSM과 Attention의 하이브리드 헤드 구조

참고:
- 논문: Hymba: A Hybrid-head Architecture for Small Language Models (arXiv:2411.13676)
- 공식 코드: https://github.com/NVlabs/hymba
- HuggingFace: https://huggingface.co/nvidia/Hymba-1.5B-Instruct
"""
from __future__ import annotations
import math
import time
import typing as T
import os
import warnings
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# ===================== FlexAttention 임포트 =====================
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        and_masks,
        or_masks,
    )
    HAS_FLEX_ATTN = True
except ImportError:
    HAS_FLEX_ATTN = False
    warnings.warn("FlexAttention을 사용할 수 없습니다. PyTorch 2.5+ 가 필요합니다.")

# Flash Attention 임포트 시도
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn("Flash Attention을 사용할 수 없습니다. 일반 SDPA를 사용합니다.")

# Mamba SSM 임포트 시도
try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    warnings.warn("Mamba SSM을 사용할 수 없습니다. Attention-only 모드로 동작합니다.")

# 환경 설정
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===================== 데이터 및 토크나이저 =====================
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC, Lowercase, Sequence as NormSeq

def get_corpus(hf_spec: str = "karpathy/tiny_shakespeare") -> str:
    """코퍼스 데이터 로드"""
    ds = load_dataset(hf_spec)
    col = "text" if "text" in ds["train"].column_names else ds["train"].column_names[0]
    return "\n\n".join(ds["train"][col])

def train_unigram(text: str, vocab_size: int = 8000, unk: str = "<|unk|>"):
    """Unigram 토크나이저 학습"""
    tk = Tokenizer(Unigram())
    tk.normalizer = NormSeq([NFKC(), Lowercase()])
    tk.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=[unk], unk_token=unk)
    tk.train_from_iterator([text], trainer=trainer)

    class Wrap:
        def __init__(self, tk):
            self.tk = tk
        def encode(self, s):
            return self.tk.encode(s).ids
        def decode(self, ids):
            return self.tk.decode(ids)
        @property
        def vocab_size(self):
            return self.tk.get_vocab_size()

    return Wrap(tk)

def make_stream_dataset(tok, text: str, seq_len: int = 512) -> TensorDataset:
    """스트리밍 데이터셋 생성"""
    import numpy as np
    ids = np.array(tok.encode(text), dtype=np.int64)
    if ids.size < seq_len + 1:
        raise RuntimeError("텍스트가 너무 짧습니다")
    x = ids[:-1]
    y = ids[1:]
    n = (len(y) // seq_len) * seq_len
    X = torch.tensor(x[:n].reshape(-1, seq_len))
    Y = torch.tensor(y[:n].reshape(-1, seq_len))
    return TensorDataset(X, Y)

def build_dataloaders(tok, text: str, seq_len: int = 512, bs: int = 32, workers: int = 0, pin: bool = True):
    """학습/검증 데이터로더 생성"""
    ds_full = make_stream_dataset(tok, text, seq_len)
    tr_len = int(0.95 * len(ds_full))
    va_len = len(ds_full) - tr_len
    tr, va = random_split(ds_full, [tr_len, va_len])
    train_dl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin)
    val_dl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_dl, val_dl

# ===================== 기본 레이어 =====================
class RMSNorm(nn.Module):
    """RMS 정규화 (Root Mean Square Normalization)

    표준편차 대신 RMS를 사용한 정규화로, LayerNorm보다 계산 효율적
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class SwiGLU(nn.Module):
    """SwiGLU 활성화 함수를 사용하는 FFN

    Swish (SiLU) 활성화와 Gating을 결합한 구조
    공식: SwiGLU(x) = Swish(W1·x) ⊙ (W2·x) → W3
    """
    def __init__(self, d: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        h = int(d * mult)
        self.w1 = nn.Linear(d, h, bias=False)  # Gate projection
        self.w2 = nn.Linear(d, h, bias=False)  # Value projection
        self.w3 = nn.Linear(h, d, bias=False)  # Output projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))

# ===================== RoPE (Rotary Position Embedding) =====================
def rotate_half(x):
    """회전을 위해 텐서의 절반을 회전"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """회전 위치 임베딩 (RoPE)

    절대 위치 임베딩 대신 상대적 위치를 회전 행렬로 인코딩
    장점: 외삽 성능이 우수하고 위치 정보가 자연스럽게 감쇠
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("_inv", inv, persistent=False)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

    def _build(self, L: int, device, dtype):
        """캐시 구축 (필요시에만)"""
        if self._cos is not None and self._cos.size(0) >= L:
            return
        t = torch.arange(L, device=device, dtype=self._inv.dtype)
        freqs = torch.einsum("i,j->ij", t, self._inv)
        self._cos = torch.cos(freqs).to(dtype)
        self._sin = torch.sin(freqs).to(dtype)

    def apply_rotary(self, x: torch.Tensor, pos: torch.Tensor):
        """회전 임베딩 적용

        Args:
            x: (B, H, T, Dh) 형태의 쿼리 또는 키 텐서
            pos: (T,) 위치 인덱스

        Returns:
            회전이 적용된 텐서
        """
        self._build(int(pos.max().item()) + 1, x.device, x.dtype)
        cos = self._cos.index_select(0, pos)[None, None, :, :]  # (1, 1, T, Dh/2)
        sin = self._sin.index_select(0, pos)[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        return torch.stack([o1, o2], dim=-1).flatten(-2)

# ===================== GQA 헬퍼 함수 =====================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA를 위한 KV 헤드 반복

    Grouped Query Attention에서 적은 수의 KV 헤드를 여러 Query 헤드가 공유
    (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

# ===================== FlexAttention 레이어 =====================
class HymbaFlexAttention(nn.Module):
    """공식 Hymba FlexAttention 구현

    특징:
    - 메타 토큰: 전역 컨텍스트를 저장하는 학습 가능한 토큰
    - Sliding Window: 로컬 어텐션으로 메모리 효율성 향상
    - FlexAttention: 동적 마스크 생성으로 유연한 어텐션 패턴
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        n_kv: int,
        local: bool = False,
        window: int = 256,
        dropout: float = 0.0,
        num_meta_tokens: int = 128,  # 공식 구현과 동일하게 128개 사용
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d % n_heads == 0
        self.H = n_heads
        self.KV = n_kv
        self.Dh = d // n_heads
        self.rep = self.H // self.KV

        self.local = local
        self.window = window
        self.num_meta_tokens = num_meta_tokens

        # Q, K, V 프로젝션
        self.q = nn.Linear(d, n_heads * self.Dh, bias=False)
        self.k = nn.Linear(d, n_kv * self.Dh, bias=False)
        self.v = nn.Linear(d, n_kv * self.Dh, bias=False)
        self.o = nn.Linear(n_heads * self.Dh, d, bias=False)

        self.rope = RotaryEmbedding(self.Dh, base=rope_base)
        self.drop = nn.Dropout(dropout)

        # FlexAttention 설정
        if HAS_FLEX_ATTN and self.local:
            self._setup_flex_attention()
        else:
            self.use_flex_attn = False

    def _setup_flex_attention(self) -> None:
        """FlexAttention 마스크 설정

        마스크 구성:
        1. 메타 토큰 (처음 M개): 모든 토큰이 볼 수 있음
        2. 일반 토큰: Causal + Sliding Window
        """
        def causal_mask(b, h, q_idx, kv_idx):
            """인과적 마스크: 미래 토큰을 볼 수 없음"""
            return q_idx >= kv_idx

        def sliding_window(b, h, q_idx, kv_idx):
            """슬라이딩 윈도우: 최근 W개 토큰만 참조"""
            return q_idx - kv_idx < self.window

        def meta_token_mask(b, h, q_idx, kv_idx):
            """메타 토큰 마스크: 처음 M개 토큰은 항상 볼 수 있음"""
            return kv_idx < self.num_meta_tokens

        # 일반 토큰: Causal + Sliding Window
        content_mask = and_masks(causal_mask, sliding_window)

        # 전체 마스크: 메타 토큰 OR (Causal + SW)
        self.attn_mask = or_masks(meta_token_mask, content_mask)
        self.create_block_mask = create_block_mask

        # FlexAttention 컴파일 (성능 최적화)
        self.flex_attention = torch.compile(flex_attention)
        self.use_flex_attn = True

    def _get_flex_block_mask(self, q_len: int, kv_len: int):
        """FlexAttention용 블록 마스크 생성"""
        block_mask = self.create_block_mask(
            self.attn_mask,
            B=None, H=None,
            Q_LEN=q_len, KV_LEN=kv_len
        )
        return block_mask

    def _make_causal_mask(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """인과적 마스크 생성

        모든 토큰(메타 + 일반)에 대해 인과적 제약 적용
        위치 i는 위치 <= i만 참조 가능
        """
        mask = torch.triu(
            torch.full((q_len, k_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask

    def _apply_sliding_window(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sliding Window Attention 적용

        메모리 효율을 위해 최근 window 크기만큼의 토큰만 유지
        메타 토큰은 항상 유지
        """
        if not self.local:
            return k, v

        Tk = k.size(2)

        if self.num_meta_tokens > 0:
            # 메타 토큰과 일반 토큰 분리
            meta_k = k[:, :, :self.num_meta_tokens, :]
            meta_v = v[:, :, :self.num_meta_tokens, :]
            content_k = k[:, :, self.num_meta_tokens:, :]
            content_v = v[:, :, self.num_meta_tokens:, :]

            # 윈도우 적용
            content_len = content_k.size(2)
            w = min(self.window, content_len)
            content_k = content_k[:, :, -w:, :]
            content_v = content_v[:, :, -w:, :]

            # 재결합
            k = torch.cat([meta_k, content_k], dim=2)
            v = torch.cat([meta_v, content_v], dim=2)
        else:
            # 메타 토큰 없음: 단순 윈도우
            w = min(self.window, Tk)
            k = k[:, :, -w:, :]
            v = v[:, :, -w:, :]

        return k, v

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_attn: bool = False,
    ):
        """
        전방 전파

        Args:
            x: 입력 텐서 [B, T, d]
            kv_cache: KV 캐시 (추론 시)
            return_attn: 어텐션 가중치 반환 여부

        Returns:
            출력 텐서와 새로운 KV 캐시
        """
        B, T, C = x.shape

        # Q, K, V 계산
        q = self.q(x).view(B, T, self.H, self.Dh).transpose(1, 2)  # (B, H, T, Dh)
        k_new = self.k(x).view(B, T, self.KV, self.Dh).transpose(1, 2)  # (B, KV, T, Dh)
        v_new = self.v(x).view(B, T, self.KV, self.Dh).transpose(1, 2)

        # KV 캐시 결합
        if kv_cache is not None and kv_cache[0] is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k_new], dim=2)
            v = torch.cat([v_prev, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        Tc = k.size(2)

        # RoPE 적용
        pos_q = torch.arange(Tc - T, Tc, device=x.device)
        pos_k = torch.arange(Tc, device=x.device)

        q = self.rope.apply_rotary(q, pos_q)
        k = self.rope.apply_rotary(k, pos_k)

        # 새 캐시 저장
        new_cache = (k.detach(), v.detach())

        # GQA: KV 헤드 반복
        k_full = repeat_kv(k, self.rep)
        v_full = repeat_kv(v, self.rep)

        # Sliding Window 적용
        k_full, v_full = self._apply_sliding_window(k_full, v_full, T)
        Tk = k_full.size(2)

        # Attention 계산
        if self.use_flex_attn and not return_attn:
            # FlexAttention 사용
            block_mask = self._get_flex_block_mask(T, Tk)
            out = self.flex_attention(q, k_full, v_full, block_mask=block_mask)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
            attn = None

        elif HAS_FLASH_ATTN and not return_attn and not self.use_flex_attn:
            # Flash Attention 사용
            q_flash = q.transpose(1, 2)  # (B, T, H, Dh)
            k_flash = k_full.transpose(1, 2)
            v_flash = v_full.transpose(1, 2)

            if self.local:
                out = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    causal=True,
                    window_size=(self.window, self.window),
                )
            else:
                out = flash_attn_func(q_flash, k_flash, v_flash, causal=True)

            out = out.reshape(B, T, self.H * self.Dh)
            attn = None

        else:
            # Manual SDPA (시각화 또는 폴백용)
            scale = 1.0 / math.sqrt(self.Dh)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale

            if Tk > 1:
                mask = self._make_causal_mask(T, Tk, x.device)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)

            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn) if self.training else attn
            out = torch.matmul(attn, v_full)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)

        out = self.o(out)

        if return_attn:
            return out, new_cache, attn
        return out, new_cache

# ===================== Mamba SSM 레이어 =====================
class HymbaMambaLayer(nn.Module):
    """Mamba SSM (State Space Model) 레이어

    상태 공간 모델을 사용한 시퀀스 모델링
    Attention에 비해 메모리 효율적 (상수 크기 상태)
    """
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if HAS_MAMBA:
            self.mamba = MambaSSM(
                d_model=d,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            # Mamba 없으면 Identity (Attention만 사용)
            self.mamba = nn.Identity()
        self.has_mamba = HAS_MAMBA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_mamba:
            return self.mamba(x)
        return x

# ===================== Hybrid Block =====================
class HymbaHybridBlock(nn.Module):
    """하이브리드 블록: Attention + Mamba SSM

    공식 Hymba 아키텍처의 핵심 구성 요소
    - Attention Head: 고해상도 리콜 (장거리 의존성)
    - Mamba Head: 효율적 컨텍스트 요약 (상수 메모리)
    - 학습 가능한 게이트로 두 경로 융합
    """

    def __init__(
        self,
        d_model: int,
        attn_dim: int,
        mamba_dim: int,
        n_heads: int,
        n_kv: int,
        local: bool = False,
        window: int = 256,
        dropout: float = 0.0,
        num_meta_tokens: int = 128,
        fusion: str = "learned_gate",  # "learned_gate", "concat", "mean"
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)

        self.attn_dim = attn_dim
        self.mamba_dim = mamba_dim
        self.fusion = fusion

        # Attention 경로
        if attn_dim > 0:
            self.to_attn = nn.Linear(d_model, attn_dim, bias=False)
            self.attn = HymbaFlexAttention(
                attn_dim, n_heads, n_kv, local, window, dropout, num_meta_tokens
            )
            if fusion != "concat":
                self.proj_attn = nn.Linear(attn_dim, d_model, bias=False)
        else:
            self.to_attn = None
            self.attn = None

        # Mamba 경로
        if mamba_dim > 0 and HAS_MAMBA:
            self.to_mamba = nn.Linear(d_model, mamba_dim, bias=False)
            self.mamba = HymbaMambaLayer(mamba_dim)
            if fusion != "concat":
                self.proj_mamba = nn.Linear(mamba_dim, d_model, bias=False)
        else:
            self.to_mamba = None
            self.mamba = None

        # Fusion 전략
        if fusion == "concat":
            self.mix = nn.Linear(attn_dim + mamba_dim, d_model, bias=False)
        elif fusion == "learned_gate":
            # 학습 가능한 게이팅 (공식 구현)
            self.gate = nn.Parameter(torch.tensor(0.5))

        # FFN
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_attn: bool = False
    ):
        """
        하이브리드 블록 전방 전파

        Args:
            x: 입력 [B, T, D]
            kv_cache: KV 캐시
            return_attn: 어텐션 가중치 반환 여부
        """
        h = self.norm1(x)

        outputs = []
        new_cache = None
        attn_weights = None

        # Attention 경로
        if self.attn is not None:
            attn_in = self.to_attn(h)
            if return_attn:
                attn_out, new_cache, attn_weights = self.attn(attn_in, kv_cache, return_attn=True)
            else:
                attn_out, new_cache = self.attn(attn_in, kv_cache, return_attn=False)
            outputs.append(("attn", attn_out))

        # Mamba 경로
        if self.mamba is not None:
            mamba_in = self.to_mamba(h)
            mamba_out = self.mamba(mamba_in)
            outputs.append(("mamba", mamba_out))

        # Fusion
        if len(outputs) == 2:
            if self.fusion == "concat":
                y = self.mix(torch.cat([outputs[0][1], outputs[1][1]], dim=-1))
            elif self.fusion == "learned_gate":
                # Sigmoid로 [0, 1] 범위로 제한
                g = torch.sigmoid(self.gate)
                y = g * self.proj_attn(outputs[0][1]) + (1 - g) * self.proj_mamba(outputs[1][1])
            else:  # mean
                y = 0.5 * self.proj_attn(outputs[0][1]) + 0.5 * self.proj_mamba(outputs[1][1])
        else:
            tag, out = outputs[0]
            if self.fusion == "concat":
                y = self.mix(out)
            else:
                y = self.proj_attn(out) if tag == "attn" else self.proj_mamba(out)

        # 잔차 연결
        x = x + self.drop(y)

        # FFN
        x = x + self.drop(self.ffn(self.norm2(x)))

        if return_attn:
            return x, new_cache, attn_weights
        return x, new_cache

# ===================== 모델 설정 =====================
@dataclass
class HymbaConfig:
    """Hymba 모델 설정

    공식 구현 기반 기본값:
    - 메타 토큰: 128개 (attention sink 방지)
    - SWA 윈도우: 256 (메모리 효율성)
    - 하이브리드 헤드: Attention + Mamba
    """
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA

    # 하이브리드 차원
    attn_dim: int = 256
    mamba_dim: int = 256

    # 메타 토큰 (공식: 128개)
    num_meta_tokens: int = 128

    # SWA 설정
    swa_layers: T.Tuple[int, ...] = (1, 2, 3, 4, 5, 7, 8, 9, 10)  # Global: 0, 6, 11
    swa_window: int = 256

    # KV 공유 그룹 (연속된 레이어끼리 공유)
    kv_share_groups: T.Tuple[T.Tuple[int, ...], ...] = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))

    # 기타
    dropout: float = 0.0
    seq_len: int = 512
    fusion: str = "learned_gate"  # Hybrid fusion 전략

# ===================== Hymba 모델 =====================
class HymbaModel(nn.Module):
    """Hymba: 하이브리드 헤드 아키텍처

    주요 특징:
    1. 메타 토큰 (128개): Attention sink 방지, 도메인 지식 저장
    2. 하이브리드 헤드: Attention (고해상도) + Mamba (효율성)
    3. Cross-layer KV 공유: 메모리 11.67배 절감
    4. Sliding Window: 장거리 의존성 유지하면서 효율적
    """

    def __init__(self, cfg: HymbaConfig):
        super().__init__()
        self.cfg = cfg
        self.swa_layers = set(cfg.swa_layers)

        # 메타 토큰 (학습 가능한 임베딩)
        if cfg.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(1, cfg.num_meta_tokens, cfg.d_model))
        else:
            self.meta_tokens = None

        # 토큰 임베딩
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # KV 공유 설정
        self.owner = [i for i in range(cfg.n_layers)]
        self.kv_group_id = [0] * cfg.n_layers

        swa = set(self.swa_layers)
        gid = -1
        i = 0
        N = cfg.n_layers

        # KV 공유 그룹 할당
        while i < N:
            is_local = i in swa

            if is_local:
                # SWA 레이어: 연속된 레이어끼리 페어링
                j = i
                while j < N and (j in swa):
                    j += 1

                k = i
                while k < j:
                    if k + 1 < j:
                        # 페어 생성
                        gid += 1
                        self.kv_group_id[k] = gid
                        self.kv_group_id[k + 1] = gid
                        self.owner[k] = k
                        self.owner[k + 1] = k  # k+1이 k의 캐시 사용
                        k += 2
                    else:
                        # 홀수 개: 마지막은 독립
                        gid += 1
                        self.kv_group_id[k] = gid
                        self.owner[k] = k
                        k += 1
                i = j
            else:
                # Global 레이어: 독립
                gid += 1
                self.kv_group_id[i] = gid
                self.owner[i] = i
                i += 1

        # 하이브리드 블록 생성
        self.blocks = nn.ModuleList()
        for li in range(cfg.n_layers):
            is_local = (li in self.swa_layers)
            self.blocks.append(HymbaHybridBlock(
                cfg.d_model, cfg.attn_dim, cfg.mamba_dim,
                cfg.n_heads, cfg.n_kv_heads,
                local=is_local, window=cfg.swa_window,
                dropout=cfg.dropout, num_meta_tokens=cfg.num_meta_tokens,
                fusion=cfg.fusion,
            ))

        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, return_attn=False):
        """
        전방 전파 (학습 모드)

        Args:
            x: 입력 토큰 ID [B, T]
            targets: 타겟 토큰 (손실 계산용)
            return_attn: 어텐션 가중치 반환
        """
        B, T = x.shape

        # 토큰 임베딩
        h = self.tok_emb(x)

        # 메타 토큰 추가
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)  # (B, M+T, D)

        # 블록 통과
        attn_weights_list = []
        for li, block in enumerate(self.blocks):
            if return_attn:
                h, _, attn_w = block(h, kv_cache=None, return_attn=True)
                attn_weights_list.append(attn_w)
            else:
                h, _ = block(h, kv_cache=None, return_attn=False)

        # 메타 토큰 제거
        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:, :]

        # 출력
        h = self.norm(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        out = {"logits": logits, "loss": loss}
        if return_attn:
            out["attn_weights"] = attn_weights_list

        return out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_kv_cache=True):
        """
        자동회귀 생성

        Args:
            idx: 입력 토큰 [B, T]
            max_new_tokens: 생성할 토큰 수
            temperature: 샘플링 온도
            top_k: Top-k 샘플링
            use_kv_cache: KV 캐시 사용 여부
        """
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)

        if not use_kv_cache:
            # 캐시 없이 생성 (간단하지만 느림)
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.cfg.seq_len else idx[:, -self.cfg.seq_len:]
                logits = self(idx_cond)["logits"]
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)

            return idx

        # KV 캐시 사용 생성
        B = idx.size(0)
        h = self.tok_emb(idx)

        # 메타 토큰 추가
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)

        # Prefill
        kv_caches = {}
        for li, block in enumerate(self.blocks):
            owner_id = self.owner[li]
            kv_cache = kv_caches.get(owner_id, None)

            h, new_cache = block(h, kv_cache, return_attn=False)

            if li == owner_id:
                kv_caches[owner_id] = new_cache

        # 메타 토큰 제거
        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:, :]

        h = self.norm(h)
        logits = self.head(h)[:, -1, :]

        # 첫 토큰 샘플링
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

        # 자동회귀 루프
        for _ in range(max_new_tokens - 1):
            h = self.tok_emb(next_token)

            for li, block in enumerate(self.blocks):
                owner_id = self.owner[li]
                kv_cache = kv_caches.get(owner_id, None)

                h, new_cache = block(h, kv_cache, return_attn=False)

                if li == owner_id:
                    kv_caches[owner_id] = new_cache

            h = self.norm(h)
            logits = self.head(h)[:, -1, :]

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def layer_table(self):
        """레이어 정보 테이블 생성"""
        import pandas as pd
        data = []
        for li in range(self.cfg.n_layers):
            is_local = li in self.swa_layers
            attn_type = "LOCAL(SWA)" if is_local else "GLOBAL"
            data.append({
                "layer": li,
                "attn": attn_type,
                "kv_owner": self.owner[li],
                "kv_share_group": self.kv_group_id[li],
            })
        return pd.DataFrame(data)

# ===================== 학습 =====================
@dataclass
class TrainConfig:
    """학습 설정"""
    seq_len: int = 512
    batch_size: int = 32
    steps: int = 10000
    lr: float = 6e-4
    warmup: int = 2000
    amp: bool = True
    grad_clip: float = 1.0

def train_loop(model, train_dl, val_dl, cfg: TrainConfig, device="cuda"):
    """학습 루프"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_schedule(step):
        if step < cfg.warmup:
            return step / cfg.warmup
        return 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup) / (cfg.steps - cfg.warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.startswith("cuda")) else None

    step = 0
    train_iter = iter(train_dl)
    t0 = time.time()

    while step < cfg.steps:
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)

        if scaler:
            with torch.amp.autocast("cuda"):
                out = model(xb, targets=yb)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            out = model(xb, targets=yb)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        opt.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        if step % 50 == 0 or step == 1:
            print(f"[{step:5d}] loss={loss.item():.3f} lr={scheduler.get_last_lr()[0]:.2e}")

    # 검증
    model.eval()
    val_loss = 0.0
    val_tokens = 0

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, targets=yb)
            val_loss += out["loss"].item() * xb.numel()
            val_tokens += xb.numel()

    val_loss /= val_tokens
    ppl = math.exp(val_loss)

    elapsed = time.time() - t0
    tps = int((step * cfg.batch_size * cfg.seq_len) / elapsed)

    print(f"\n=== 학습 완료 ===")
    print(f"Steps: {step}, 시간: {elapsed/60:.1f}분")
    print(f"Train loss: {loss.item():.3f}, Val loss: {val_loss:.3f}, PPL: {ppl:.2f}")
    print(f"처리량: {tps:,} tokens/s")

    return {
        "train_loss": loss.item(),
        "val_loss": val_loss,
        "ppl": ppl,
        "tps": tps
    }

def build_everything(seq_len: int = 512, bs: int = 32, vocab_size: int = 8000):
    """코퍼스, 토크나이저, 데이터로더 생성"""
    print("코퍼스 로딩 중...")
    corpus = get_corpus()

    print(f"토크나이저 학습 중 (vocab_size={vocab_size})...")
    tok = train_unigram(corpus, vocab_size=vocab_size)

    print(f"데이터로더 생성 중 (seq_len={seq_len}, bs={bs})...")
    train_dl, val_dl = build_dataloaders(tok, corpus, seq_len=seq_len, bs=bs)

    return corpus, tok, train_dl, val_dl
