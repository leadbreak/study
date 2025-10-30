"""
Hymba v3: Improved implementation with flash_attn and flex_attention support

Key improvements over v2: 공식 코드에서 사용한 transformers 및 torch의 flex attention 모듈을 활용하여 성능 및 유연성 향상
1. Flash Attention 2 integration for better performance
2. Flex Attention support for meta tokens + SWA (PyTorch >= 2.5.0)
3. Proper meta token handling in attention masks
4. Better RoPE implementation compatible with transformers
5. Optimized KV cache management
"""
from __future__ import annotations
import math, time, typing as T, os, warnings, inspect
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Try to import flash_attn
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn("flash_attn not available, falling back to PyTorch SDPA")

# Try to import flex_attention for meta tokens + SWA
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks
    HAS_FLEX_ATTN = True
except ImportError:
    HAS_FLEX_ATTN = False

# env / warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===================== Data / Tokenizer =====================
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC, Lowercase, Sequence as NormSeq

def get_corpus(hf_spec:str="karpathy/tiny_shakespeare") -> str:
    ds = load_dataset(hf_spec)
    col = "text" if "text" in ds["train"].column_names else ds["train"].column_names[0]
    return "\n\n".join(ds["train"][col])

def train_unigram(text:str, vocab_size:int=8000, unk:str="<|unk|>"):
    tk = Tokenizer(Unigram())
    tk.normalizer = NormSeq([NFKC(), Lowercase()])
    tk.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=[unk], unk_token=unk)
    tk.train_from_iterator([text], trainer=trainer)

    class Wrap:
        def __init__(self, tk): self.tk=tk
        def encode(self, s): return self.tk.encode(s).ids
        def decode(self, ids): return self.tk.decode(ids)
        @property
        def vocab_size(self): return self.tk.get_vocab_size()
    return Wrap(tk)

def make_stream_dataset(tok, text:str, seq_len:int=512) -> TensorDataset:
    import numpy as np
    ids = np.array(tok.encode(text), dtype=np.int64)
    if ids.size < seq_len+1: raise RuntimeError("Text too short")
    x = ids[:-1]; y = ids[1:]
    n = (len(y)//seq_len)*seq_len
    X = torch.tensor(x[:n].reshape(-1, seq_len))
    Y = torch.tensor(y[:n].reshape(-1, seq_len))
    return TensorDataset(X,Y)

def build_dataloaders(tok, text:str, seq_len:int=512, bs:int=32, workers:int=0, pin:bool=True):
    ds_full = make_stream_dataset(tok, text, seq_len)
    tr_len = int(0.95*len(ds_full)); va_len = len(ds_full)-tr_len
    tr, va = random_split(ds_full, [tr_len, va_len])
    train_dl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin)
    val_dl   = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_dl, val_dl

# ===================== Layers =====================
class RMSNorm(nn.Module):
    """RMS Normalization (same as transformers HymbaRMSNorm)"""
    def __init__(self, d:int, eps:float=1e-6):
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
    def __init__(self, d:int, mult:float=4.0, dropout:float=0.0):
        super().__init__()
        h=int(d*mult)
        self.w1=nn.Linear(d,h,bias=False); self.w2=nn.Linear(d,h,bias=False); self.w3=nn.Linear(h,d,bias=False)
        self.drop=nn.Dropout(dropout)
    def forward(self, x): return self.w3(self.drop(F.silu(self.w1(x))*self.w2(x)))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Simplified RoPE implementation"""
    def __init__(self, dim:int, base:float=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("_inv", inv, persistent=False)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

    def _build(self, L:int, device, dtype):
        if self._cos is not None and self._cos.size(0) >= L:
            return
        t = torch.arange(L, device=device, dtype=self._inv.dtype)
        freqs = torch.einsum("i,j->ij", t, self._inv)
        self._cos = torch.cos(freqs).to(dtype)
        self._sin = torch.sin(freqs).to(dtype)

    def apply_rotary(self, x:torch.Tensor, pos:torch.Tensor):
        """
        Apply rotary embedding
        Args:
            x: (B, H, T, Dh) or (B, KV, T, Dh) - works with any head dimension
            pos: (T,) position indices
        Returns:
            rotated x
        """
        self._build(int(pos.max().item())+1, x.device, x.dtype)
        cos = self._cos.index_select(0, pos)[None, None, :, :]  # (1, 1, T, Dh/2)
        sin = self._sin.index_select(0, pos)[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        o1 = x1*cos - x2*sin
        o2 = x1*sin + x2*cos
        return torch.stack([o1, o2], dim=-1).flatten(-2)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for GQA (compatible with transformers)
    (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class AttnLayerV3(nn.Module):
    """
    Improved attention layer with:
    - Flash Attention 2 support
    - Flex Attention for meta tokens + SWA
    - GQA + RoPE
    - Proper meta token masking
    - Differential Attention for global layers (noise cancellation)
    """

    # Differential Attention lambda initialization constants
    # Formula: λ_init = LAMBDA_BASE - LAMBDA_SCALE * exp(-LAMBDA_DECAY * (layer - 1))
    # These control how differential attention strength varies by layer depth
    LAMBDA_BASE = 0.8      # Base lambda value (upper bound at layer 0)
    LAMBDA_SCALE = 0.6     # Scaling factor for exponential decay
    LAMBDA_DECAY = 0.3     # Decay rate across layers

    # Lambda parameter initialization
    LAMBDA_PARAM_INIT_STD = 0.1  # Standard deviation for lambda parameter initialization

    def __init__(
        self,
        d:int,
        n_heads:int,
        n_kv:int,
        local:bool=False,
        window:int=256,
        dropout:float=0.0,
        num_meta_tokens:int=0,
        use_flash_attn:bool=True,
        use_flex_attn:bool=False,  # Enable for meta tokens + SWA
        use_differential:bool=False,  # Enable differential attention for global layers
        diff_exclude_meta:bool=False,  # Exclude meta tokens from differential computation
        layer_idx:int=0,  # Layer index for lambda_init calculation
        rope_base:float=10000.0,
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
        self.use_differential = use_differential and not local  # Only use differential for global attention
        self.diff_exclude_meta = diff_exclude_meta and self.use_differential  # Only applies to differential attention

        # Differential attention: split Q/K into two groups
        if self.use_differential:
            # Q and K are projected to double dimension, then split
            self.q = nn.Linear(d, 2 * n_heads * self.Dh, bias=False)
            self.k = nn.Linear(d, 2 * n_kv * self.Dh, bias=False)
            # V remains the same but double dimension for compatibility
            self.v = nn.Linear(d, 2 * n_kv * self.Dh, bias=False)
            self.o = nn.Linear(2 * n_heads * self.Dh, d, bias=False)

            # Lambda parameters for differential attention
            # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
            # Initialize learnable lambda parameters
            self.lambda_q1 = nn.Parameter(torch.randn(self.Dh) * self.LAMBDA_PARAM_INIT_STD)
            self.lambda_k1 = nn.Parameter(torch.randn(self.Dh) * self.LAMBDA_PARAM_INIT_STD)
            self.lambda_q2 = nn.Parameter(torch.randn(self.Dh) * self.LAMBDA_PARAM_INIT_STD)
            self.lambda_k2 = nn.Parameter(torch.randn(self.Dh) * self.LAMBDA_PARAM_INIT_STD)

            # Layer-dependent lambda initialization
            # Formula: λ_init = LAMBDA_BASE - LAMBDA_SCALE * exp(-LAMBDA_DECAY * (layer - 1))
            # This creates varying differential strength across layers
            self.lambda_init = (
                self.LAMBDA_BASE -
                self.LAMBDA_SCALE * math.exp(-self.LAMBDA_DECAY * max(0, layer_idx - 1))
            )

            # GroupNorm for multi-head normalization
            self.group_norm = nn.GroupNorm(num_groups=n_heads, num_channels=2 * n_heads * self.Dh, affine=True)
        else:
            # Standard attention
            self.q = nn.Linear(d, n_heads*self.Dh, bias=False)
            self.k = nn.Linear(d, n_kv*self.Dh, bias=False)
            self.v = nn.Linear(d, n_kv*self.Dh, bias=False)
            self.o = nn.Linear(n_heads*self.Dh, d, bias=False)

        self.rope = RotaryEmbedding(self.Dh, base=rope_base)
        self.drop = nn.Dropout(dropout)

        # Flash / Flex attention config
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN and not self.use_differential
        self.use_flex_attn = use_flex_attn and HAS_FLEX_ATTN and self.local and self.num_meta_tokens > 0

        if self.use_flex_attn:
            # Pre-compile flex attention mask
            self._setup_flex_attention()

    def _setup_flex_attention(self) -> None:
        """Setup flex attention with meta token + SWA masking"""
        if not HAS_FLEX_ATTN:
            warnings.warn("Flex attention not available, falling back to standard attention")
            self.use_flex_attn = False
            return

        def sliding_window(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= self.window

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Content tokens have causal + sliding window
        attn_mask = and_masks(causal_mask, sliding_window)

        # Meta tokens can attend to everything
        def prefix_mask(b, h, q_idx, kv_idx):
            return kv_idx < self.num_meta_tokens

        register_mask = and_masks(causal_mask, prefix_mask)

        # Combine: content uses SWA, meta tokens are always visible
        self.attn_mask = or_masks(attn_mask, register_mask)
        self.create_block_mask = create_block_mask
        self.flex_attention = torch.compile(flex_attention)

    def _get_flex_block_mask(self, q_len: int, kv_len: int):
        """
        Get or create block mask for flex attention.

        Args:
            q_len: Query sequence length
            kv_len: Key/value sequence length

        Returns:
            Block mask for flex attention
        """
        block_mask = self.create_block_mask(
            self.attn_mask,
            B=None, H=None,
            Q_LEN=q_len, KV_LEN=kv_len
        )
        return block_mask

    def _local_slice(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sliding window for SWA (only for non-flex attention).
        For meta tokens: keep all meta tokens + window of content.

        Args:
            k: Key tensor [B, H, Tk, Dh]
            v: Value tensor [B, H, Tk, Dh]
            q_seq_len: Query sequence length

        Returns:
            Sliced (k, v) tuple
        """
        if not self.local:
            return k, v

        Tk = k.size(2)

        if self.num_meta_tokens > 0:
            # Split meta and content
            meta_k = k[:, :, :self.num_meta_tokens, :]
            meta_v = v[:, :, :self.num_meta_tokens, :]
            content_k = k[:, :, self.num_meta_tokens:, :]
            content_v = v[:, :, self.num_meta_tokens:, :]

            # Apply window to content
            content_len = content_k.size(2)
            w = min(self.window, content_len)
            content_k = content_k[:, :, -w:, :]
            content_v = content_v[:, :, -w:, :]

            # Concatenate back
            k = torch.cat([meta_k, content_k], dim=2)
            v = torch.cat([meta_v, content_v], dim=2)
        else:
            # No meta tokens, simple window
            w = min(self.window, Tk)
            k = k[:, :, -w:, :]
            v = v[:, :, -w:, :]

        return k, v

    def _make_causal_mask_with_meta(
        self,
        q_len: int,
        k_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create causal mask - ALL tokens follow causal constraint.

        CORRECTED: Meta tokens are NOT bidirectional!
        - Position i can only attend to positions <= i
        - This applies to BOTH meta tokens and content tokens

        Example with 4 meta + 4 content tokens:
                 m0  m1  m2  m3  t0  t1  t2  t3
            m0 [  0  -∞  -∞  -∞  -∞  -∞  -∞  -∞ ]  ← causal
            m1 [  0   0  -∞  -∞  -∞  -∞  -∞  -∞ ]  ← causal
            m2 [  0   0   0  -∞  -∞  -∞  -∞  -∞ ]  ← causal
            m3 [  0   0   0   0  -∞  -∞  -∞  -∞ ]  ← causal
            t0 [  0   0   0   0   0  -∞  -∞  -∞ ]  ← causal
            t1 [  0   0   0   0   0   0  -∞  -∞ ]  ← causal
            t2 [  0   0   0   0   0   0   0  -∞ ]  ← causal
            t3 [  0   0   0   0   0   0   0   0 ]  ← causal
        """
        # Standard causal mask for ALL tokens (meta + content)
        mask = torch.triu(
            torch.full((q_len, k_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask

    def _apply_causal_mask(self, scores: torch.Tensor, q_len: int, kv_len: int) -> torch.Tensor:
        """
        Apply causal mask to attention scores.

        This is a helper method to avoid duplicating mask creation and application logic.
        The mask ensures that position i can only attend to positions <= i (causal constraint).

        Args:
            scores: Attention scores of shape [B, H, q_len, kv_len]
            q_len: Query sequence length
            kv_len: Key sequence length

        Returns:
            Masked scores of shape [B, H, q_len, kv_len]
        """
        mask = self._make_causal_mask_with_meta(q_len, kv_len, scores.device)
        return scores + mask.unsqueeze(0).unsqueeze(0)

    def _compute_single_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float,
        causal_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Helper to compute single attention map with causal masking.

        Args:
            q: Query tensor [B, H, T, Dh]
            k: Key tensor [B, H, Tk, Dh]
            scale: Scaling factor (1/sqrt(Dh))
            causal_mask: Optional causal mask [T, Tk]

        Returns:
            Attention weights [B, H, T, Tk]
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal_mask is not None:
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        return self.drop(attn) if self.training else attn

    def _concat_kv_cache(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to concatenate new KV with cached KV.

        Args:
            k_new: New key tensor [B, H, T, Dh]
            v_new: New value tensor [B, H, T, Dh]
            kv_cache: Optional cached (k, v) tuple

        Returns:
            Concatenated (k, v) tuple
        """
        if kv_cache is not None and kv_cache[0] is not None:
            k_prev, v_prev = kv_cache
            return torch.cat([k_prev, k_new], dim=2), torch.cat([v_prev, v_new], dim=2)
        return k_new, v_new

    def _manual_attention(
        self,
        q: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        T: int,
        Tk: int,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Helper for manual SDPA computation.

        Args:
            q: Query tensor [B, H, T, Dh]
            k_full: Key tensor [B, H, Tk, Dh]
            v_full: Value tensor [B, H, Tk, Dh]
            T: Query sequence length
            Tk: Key sequence length
            device: Device for mask creation

        Returns:
            Tuple of (output [B, T, H*Dh], attention_weights [B, H, T, Tk])
        """
        B = q.shape[0]
        scale = 1.0 / math.sqrt(self.Dh)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        if Tk > 1:
            scores = self._apply_causal_mask(scores, T, Tk)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn) if self.training else attn
        out = torch.matmul(attn, v_full)
        out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
        return self.o(out), attn

    def _compute_differential_attention(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        v: torch.Tensor,
        causal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute differential attention: DiffAttn = (softmax(Q1 K1^T) - λ softmax(Q2 K2^T)) V

        IMPORTANT: This computes two separate attention maps and combines them.
        The lambda scaling is applied to the DIFFERENCE, not to individual attention weights.

        Meta token handling:
        - If diff_exclude_meta=False: Apply differential attention to all tokens (default)
        - If diff_exclude_meta=True (CORRECTED):
          * Meta tokens use STANDARD attention (no differential)
          * Content tokens use DIFFERENTIAL attention
          * Attention map retains full size (T, Tk) with both meta and content
          * Meta tokens can interact with all tokens normally
          * Content tokens get noise cancellation benefit

        Visualization semantics:
        - If diff_exclude_meta=False: Returns differential map (attn1 - λ·attn2)
          * Can have NEGATIVE values
          * Does NOT sum to 1 across keys
          * Use for research/debugging differential mechanism

        - If diff_exclude_meta=True: Returns MIXED semantics map
          * First M rows: standard attention (sum to 1) - for meta tokens
          * Remaining rows: differential attention (can be negative) - for content tokens
          * Use with caution in visualization tools

        Args:
            q1, q2: Query tensors [B, H, T, Dh]
            k1, k2: Key tensors [B, H, Tk, Dh]
            v: Value tensor [B, H, Tk, 2*Dh]
            causal_mask: Optional causal mask [T, Tk]

        Returns:
            out: Output tensor [B, T, 2*H*Dh]
            attn_map: Attention map for visualization [B, H, T, Tk]
                - If diff_exclude_meta=False: differential map (attn1 - λ·attn2)
                - If diff_exclude_meta=True: mixed map (standard for meta, differential for content)
        """
        B, H, T, Dh = q1.shape
        Tk = k1.size(2)
        scale = 1.0 / math.sqrt(Dh)

        # Compute lambda: λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_val = lambda_1 - lambda_2 + self.lambda_init

        if self.diff_exclude_meta and self.num_meta_tokens > 0:
            # CORRECTED: Keep full attention map, but use different attention for meta vs content
            M = self.num_meta_tokens

            # Split queries into meta and content
            q1_meta = q1.narrow(2, 0, M)      # (B, H, M, Dh)
            q1_content = q1.narrow(2, M, T - M)  # (B, H, T-M, Dh)
            q2_meta = q2.narrow(2, 0, M)
            q2_content = q2.narrow(2, M, T - M)

            # For meta tokens: use standard attention (average of q1 and q2 attention)
            # This is equivalent to differential with lambda=0
            attn1_meta = self._compute_single_attention(q1_meta, k1, scale, causal_mask[:M, :] if causal_mask is not None else None)
            attn2_meta = self._compute_single_attention(q2_meta, k2, scale, causal_mask[:M, :] if causal_mask is not None else None)

            # Standard attention for meta: average of two attention maps
            attn_meta = (attn1_meta + attn2_meta) / 2.0  # (B, H, M, Tk)

            # Compute output for meta tokens
            out_meta = torch.matmul(attn_meta, v)  # (B, H, M, 2*Dh)

            # For content tokens: use differential attention
            causal_mask_content = causal_mask[M:, :] if causal_mask is not None else None
            attn1_content = self._compute_single_attention(q1_content, k1, scale, causal_mask_content)
            attn2_content = self._compute_single_attention(q2_content, k2, scale, causal_mask_content)

            # Differential attention for content
            diff_attn_content = attn1_content - lambda_val * attn2_content  # (B, H, T-M, Tk)

            # Compute output for content tokens
            out_content = torch.matmul(diff_attn_content, v)  # (B, H, T-M, 2*Dh)

            # Concatenate meta and content outputs
            out = torch.cat([out_meta, out_content], dim=2)  # (B, H, T, 2*Dh)

            # Concatenate attention maps for visualization (MIXED semantics)
            # - First M rows: standard attention (probabilities, sum to 1)
            # - Remaining rows: differential attention (can be negative, doesn't sum to 1)
            attn_map = torch.cat([attn_meta, diff_attn_content], dim=2)  # (B, H, T, Tk)

        else:
            # Standard differential attention over all tokens using helper
            attn1 = self._compute_single_attention(q1, k1, scale, causal_mask)
            attn2 = self._compute_single_attention(q2, k2, scale, causal_mask)

            # Compute differential attention map: attn1 - λ·attn2
            # Note: Can have NEGATIVE values, does NOT sum to 1
            diff_attn = attn1 - lambda_val * attn2  # (B, H, T, Tk)

            # Compute output using differential attention
            out = torch.matmul(diff_attn, v)  # (B, H, T, 2*Dh)

            # Return pure differential attention map for visualization
            attn_map = diff_attn

        # Reshape: (B, H, T, 2*Dh) -> (B, T, 2*H*Dh)
        out = out.transpose(1, 2).reshape(B, T, 2 * self.H * self.Dh)

        # GroupNorm for per-head normalization
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        # Scale by (1 - lambda_init) to align gradient flow
        out = out * (1.0 - self.lambda_init)

        # Return attention map for visualization
        # WARNING: Interpretation depends on diff_exclude_meta setting (see docstring)
        return out, attn_map

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        role: str = "owner",
        return_attn: bool = False,
    ):
        """
        Forward pass with optional KV cache and role-based computation

        Args:
            x: Input tensor [B, T, d]
            kv_cache: Cached (K, V) from owner
            role: "owner" computes and caches KV, "follower" reuses owner's KV
            return_attn: Return attention weights for visualization
        """
        B, T, C = x.shape

        # Differential attention path (only for global layers)
        if self.use_differential:
            # Q and K are projected to double dimension, then split
            q_full = self.q(x).view(B, T, self.H, 2 * self.Dh).transpose(1, 2)  # (B, H, T, 2*Dh)
            q1, q2 = torch.chunk(q_full, 2, dim=-1)  # Each: (B, H, T, Dh)

            if role == "follower":
                assert kv_cache is not None and kv_cache[0] is not None
                k_owner, v_owner = kv_cache  # (B, KV, Tc, 2*Dh)

                # Repeat KV for GQA
                k_full = repeat_kv(k_owner, self.rep)  # (B, H, Tc, 2*Dh)
                v_full = repeat_kv(v_owner, self.rep)
                Tc = k_full.size(2)

                # Split K into two groups
                k1, k2 = torch.chunk(k_full, 2, dim=-1)  # Each: (B, H, Tc, Dh)

                # RoPE for current queries
                pos_q = torch.arange(Tc - T, Tc, device=x.device)
                q1 = self.rope.apply_rotary(q1, pos_q)
                q2 = self.rope.apply_rotary(q2, pos_q)

                # Causal mask
                causal_mask = self._make_causal_mask_with_meta(T, Tc, x.device) if Tc > 1 else None

                # Compute differential attention
                out, attn1 = self._compute_differential_attention(q1, q2, k1, k2, v_full, causal_mask)
                out = self.o(out)

                if return_attn:
                    return out, None, attn1  # Return primary attention pattern for visualization
                return out, None

            # Owner path: compute and cache KV
            k_new = self.k(x).view(B, T, self.KV, 2 * self.Dh).transpose(1, 2)  # (B, KV, T, 2*Dh)
            v_new = self.v(x).view(B, T, self.KV, 2 * self.Dh).transpose(1, 2)

            # Concatenate with cache using helper
            k, v = self._concat_kv_cache(k_new, v_new, kv_cache)

            Tc = k.size(2)

            # Repeat KV for GQA
            k_full = repeat_kv(k, self.rep)  # (B, H, Tc, 2*Dh)
            v_full = repeat_kv(v, self.rep)

            # Split K into two groups
            k1, k2 = torch.chunk(k_full, 2, dim=-1)  # Each: (B, H, Tc, Dh)

            # RoPE
            pos_q = torch.arange(Tc - T, Tc, device=x.device)
            pos_k = torch.arange(Tc, device=x.device)

            q1 = self.rope.apply_rotary(q1, pos_q)
            q2 = self.rope.apply_rotary(q2, pos_q)

            # Apply RoPE to K (apply to both k1 and k2)
            k1 = self.rope.apply_rotary(k1, pos_k)
            k2 = self.rope.apply_rotary(k2, pos_k)

            # Save cache (before splitting, store the full K/V)
            new_cache = (k.detach(), v.detach())

            # Causal mask
            causal_mask = self._make_causal_mask_with_meta(T, Tc, x.device) if Tc > 1 else None

            # Compute differential attention
            out, attn1 = self._compute_differential_attention(q1, q2, k1, k2, v_full, causal_mask)
            out = self.o(out)

            if return_attn:
                return out, new_cache, attn1  # Return primary attention pattern
            return out, new_cache

        # Standard attention path (local or non-differential global)
        # Always compute Q
        q = self.q(x).view(B, T, self.H, self.Dh).transpose(1, 2)  # (B, H, T, Dh)

        if role == "follower":
            # Reuse owner's KV cache
            assert kv_cache is not None and kv_cache[0] is not None
            k_owner, v_owner = kv_cache  # (B, KV, Tc, Dh)

            # Repeat KV for GQA
            k_full = repeat_kv(k_owner, self.rep)  # (B, H, Tc, Dh)
            v_full = repeat_kv(v_owner, self.rep)
            Tc = k_full.size(2)

            # RoPE for current queries
            pos_q = torch.arange(Tc - T, Tc, device=x.device)
            q = self.rope.apply_rotary(q, pos_q)

            # Sliding window
            k_full, v_full = self._local_slice(k_full, v_full, T)
            Tk = k_full.size(2)

            # Attention
            if self.use_flex_attn and not return_attn:
                # Use flex attention (only in forward mode)
                q_len = q.size(2)
                kv_len = k_full.size(2)
                block_mask = self._get_flex_block_mask(q_len, kv_len)
                out = self.flex_attention(q, k_full, v_full, block_mask=block_mask)
                out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
                out = self.o(out)
                return out, None

            elif self.use_flash_attn and not return_attn and self.num_meta_tokens == 0:
                # Flash attention (no meta tokens, simpler)
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
                out = self.o(out)
                return out, None

            else:
                # Manual attention (for return_attn or fallback)
                scale = 1.0 / math.sqrt(self.Dh)
                scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B, H, T, Tk)

                # Causal mask with meta token support
                if Tk > 1:
                    scores = self._apply_causal_mask(scores, T, Tk)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn) if self.training else attn
                out = torch.matmul(attn, v_full)  # (B, H, T, Dh)
                out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
                out = self.o(out)

                if return_attn:
                    return out, None, attn
                return out, None

        # Owner: compute and cache KV
        k_new = self.k(x).view(B, T, self.KV, self.Dh).transpose(1, 2)  # (B, KV, T, Dh)
        v_new = self.v(x).view(B, T, self.KV, self.Dh).transpose(1, 2)

        # Concatenate with cache using helper
        k, v = self._concat_kv_cache(k_new, v_new, kv_cache)

        Tc = k.size(2)

        # RoPE
        pos_q = torch.arange(Tc - T, Tc, device=x.device)
        pos_k = torch.arange(Tc, device=x.device)

        q = self.rope.apply_rotary(q, pos_q)

        # Apply RoPE directly to K (B, KV, Tc, Dh)
        k = self.rope.apply_rotary(k, pos_k)

        # Save cache
        new_cache = (k.detach(), v.detach())

        # GQA repeat
        k_full = repeat_kv(k, self.rep)
        v_full = repeat_kv(v, self.rep)

        # SDPA with proper SWA handling
        if return_attn and self.local:
            # For SWA visualization: compute per-query windows to show true SWA pattern
            # Based on hymba_v2 implementation (lines 273-324)
            scale = 1.0 / math.sqrt(self.Dh)
            full_attn = torch.zeros(B, self.H, T, Tc, device=x.device, dtype=k_full.dtype)

            for t in range(T):
                # Absolute position of this query in the full sequence
                q_abs = Tc - T + t

                # SWA window for CONTENT tokens (excluding meta)
                # Window covers [q_abs - window + 1, q_abs] among CONTENT positions
                # But we need to map this to absolute positions INCLUDING meta tokens

                # Content positions start at num_meta_tokens
                # So absolute position q_abs maps to content position (q_abs - num_meta_tokens)

                if self.num_meta_tokens > 0:
                    # Meta tokens are ALWAYS included: positions [0:num_meta_tokens]
                    # For content, calculate window
                    content_start_abs = self.num_meta_tokens

                    # Window start for content (in absolute coordinates)
                    # We want recent `window` content tokens up to and including current position
                    window_start_content = max(content_start_abs, q_abs - self.window + 1)
                    window_end = q_abs + 1  # Inclusive of current position

                    # Build k/v: [meta tokens] + [content window]
                    k_meta = k_full[:, :, :self.num_meta_tokens, :]  # (B, H, M, Dh)
                    v_meta = v_full[:, :, :self.num_meta_tokens, :]

                    k_content_window = k_full[:, :, window_start_content:window_end, :]  # (B, H, W, Dh)
                    v_content_window = v_full[:, :, window_start_content:window_end, :]

                    # Concatenate
                    k_window = torch.cat([k_meta, k_content_window], dim=2)
                    v_window = torch.cat([v_meta, v_content_window], dim=2)

                    # Key indices: meta [0:M] + content [window_start:window_end]
                    key_indices = list(range(self.num_meta_tokens)) + list(range(window_start_content, window_end))
                else:
                    # No meta tokens - standard sliding window
                    window_start = max(0, q_abs - self.window + 1)
                    window_end = q_abs + 1

                    k_window = k_full[:, :, window_start:window_end, :]
                    v_window = v_full[:, :, window_start:window_end, :]
                    key_indices = list(range(window_start, window_end))

                # Compute attention for this query
                q_single = q[:, :, t:t+1, :]  # (B, H, 1, Dh)
                scores = torch.matmul(q_single, k_window.transpose(-2, -1)) * scale  # (B, H, 1, total_keys)

                attn_window = F.softmax(scores, dim=-1)
                attn_window = self.drop(attn_window) if self.training else attn_window

                # Place in full attention matrix at correct positions
                for i, k_idx in enumerate(key_indices):
                    full_attn[:, :, t, k_idx] = attn_window[:, :, 0, i]

            # Compute output using full attention
            out = torch.matmul(full_attn, v_full)  # (B, H, T, Dh)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
            out = self.o(out)

            return out, new_cache, full_attn

        elif return_attn:
            # Global attention with return_attn (non-local case)
            scale = 1.0 / math.sqrt(self.Dh)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B, H, T, Tc)
            if Tc > 1:
                # Use meta-aware causal mask for consistency
                scores = self._apply_causal_mask(scores, T, Tc)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn) if self.training else attn
            out = torch.matmul(attn, v_full)  # (B, H, T, Dh)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
            out = self.o(out)

            return out, new_cache, attn

        else:
            # Training path: use efficient shared window for SWA
            k_full, v_full = self._local_slice(k_full, v_full, T)
            Tk = k_full.size(2)

            if self.use_flex_attn:
                q_len = q.size(2)
                kv_len = k_full.size(2)
                block_mask = self._get_flex_block_mask(q_len, kv_len)
                out = self.flex_attention(q, k_full, v_full, block_mask=block_mask)
                out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
                out = self.o(out)
                return out, new_cache

            elif self.use_flash_attn and self.num_meta_tokens == 0:
                q_flash = q.transpose(1, 2)
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
                out = self.o(out)
                return out, new_cache

            else:
                # Manual SDPA
                scale = 1.0 / math.sqrt(self.Dh)
                scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale

                if Tk > 1:
                    scores = self._apply_causal_mask(scores, T, Tk)

                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn) if self.training else attn
                out = torch.matmul(attn, v_full)
                out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
                out = self.o(out)

                return out, new_cache


class BlockV3(nn.Module):
    """Transformer block with improved attention"""
    def __init__(
        self,
        d:int,
        n_heads:int,
        n_kv:int,
        local:bool=False,
        window:int=256,
        dropout:float=0.0,
        num_meta_tokens:int=0,
        use_flash_attn:bool=True,
        use_flex_attn:bool=False,
        use_differential:bool=False,
        diff_exclude_meta:bool=False,
        layer_idx:int=0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = AttnLayerV3(
            d, n_heads, n_kv, local, window, dropout, num_meta_tokens,
            use_flash_attn=use_flash_attn,
            use_flex_attn=use_flex_attn,
            use_differential=use_differential,
            diff_exclude_meta=diff_exclude_meta,
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, dropout=dropout)

    def forward(self, x, kv_cache=None, role="owner", return_attn=False):
        """
        Block forward pass

        Args:
            x: Input tensor (B, T, D)
            kv_cache: KV cache for inference (None during training)
            role: "owner" or "follower" for KV sharing (only used during inference)
            return_attn: If True, return attention weights

        Returns:
            x: Output tensor (B, T, D)
            cache/attn: KV cache (inference) or attention weights (visualization)
        """
        # Attention with residual
        h = self.norm1(x)

        # CORRECTED: No special handling needed at Block level
        # The attention layer now returns full output including meta tokens
        if return_attn:
            attn_out, new_cache, attn_weights = self.attn(h, kv_cache, role, return_attn=True)
            x = x + attn_out
            # FFN with residual
            x = x + self.ffn(self.norm2(x))
            return x, attn_weights
        else:
            attn_out, new_cache = self.attn(h, kv_cache, role, return_attn=False)
            x = x + attn_out
            # FFN with residual
            x = x + self.ffn(self.norm2(x))
            return x, new_cache


@dataclass
class ModelCfgV3:
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    dropout: float = 0.0
    seq_len: int = 512
    swa_layers: T.Tuple[int,...] = (1,2,3,4,5,7,8,9,10)
    swa_window: int = 256
    num_meta_tokens: int = 4
    kv_share_groups: T.Tuple[T.Tuple[int,...],...] = ((1,2), (3,4), (5,6), (7,8), (9,10))
    use_flash_attn: bool = True  # Use flash attention if available
    use_flex_attn: bool = False  # Use flex attention for meta + SWA
    use_differential: bool = True  # Use differential attention for global layers (noise cancellation)
    diff_exclude_meta: bool = False  # If True, exclude meta tokens from differential attention computation


class HymbaV3(nn.Module):
    """Hymba v3 with flash_attn and flex_attention support"""
    def __init__(self, cfg: ModelCfgV3):
        super().__init__()
        self.cfg = cfg
        self.swa_layers = set(cfg.swa_layers)

        # Meta tokens
        if cfg.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(1, cfg.num_meta_tokens, cfg.d_model))
        else:
            self.meta_tokens = None

        # Token embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # KV sharing setup - EXACTLY matching hymba_v2 logic
        # Rules:
        # 1. Consecutive SWA layers are paired (2 layers share KV: owner is first of pair)
        # 2. Global layers are independent (owner = self)
        # 3. Differential layers are independent (dimension mismatch: 2*Dh vs Dh)
        # 4. Group IDs increment sequentially

        self.owner = [i for i in range(cfg.n_layers)]  # Default: each is own owner
        self.kv_group_id = [0] * cfg.n_layers  # Will be filled

        swa = set(self.swa_layers)
        gid = -1  # Group ID counter
        i = 0
        N = cfg.n_layers

        while i < N:
            is_local = i in swa
            use_diff = cfg.use_differential and not is_local

            if is_local and not use_diff:  # SWA layer without differential
                # Find consecutive SWA layers
                j = i
                while j < N and (j in swa):
                    # Check if this SWA layer uses differential
                    layer_use_diff = cfg.use_differential and not (j in swa)
                    if layer_use_diff:
                        break
                    j += 1

                # Process SWA block [i, j)
                k = i
                while k < j:
                    if k + 1 < j:
                        # Pair two consecutive SWA layers
                        gid += 1
                        self.kv_group_id[k] = gid
                        self.kv_group_id[k + 1] = gid
                        self.owner[k] = k  # First is owner
                        self.owner[k + 1] = k  # Second uses first's cache
                        k += 2
                    else:
                        # Single SWA layer (odd one out)
                        gid += 1
                        self.kv_group_id[k] = gid
                        self.owner[k] = k
                        k += 1
                i = j
            else:
                # Global layer or differential layer: independent
                gid += 1
                self.kv_group_id[i] = gid
                self.owner[i] = i
                i += 1

        # Build blocks
        self.blocks = nn.ModuleList()
        for li in range(cfg.n_layers):
            is_local = (li in self.swa_layers)
            # Use differential attention only for global (non-local) layers
            use_diff = cfg.use_differential and not is_local
            self.blocks.append(BlockV3(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                local=is_local, window=cfg.swa_window,
                dropout=cfg.dropout, num_meta_tokens=cfg.num_meta_tokens,
                use_flash_attn=cfg.use_flash_attn,
                use_flex_attn=cfg.use_flex_attn,
                use_differential=use_diff,
                diff_exclude_meta=cfg.diff_exclude_meta,
                layer_idx=li,
            ))

        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, return_attn=False):
        """
        Forward pass for training/evaluation (teacher forcing)

        Args:
            x: Input token IDs (B, T)
            targets: Target token IDs for loss computation (B, T)
            return_attn: If True, return attention weights for visualization

        Note:
            - During training (targets provided), KV cache is NOT used
            - Meta tokens are prepended to input and removed before loss calculation
            - Attention weights include meta tokens when return_attn=True
        """
        B, T = x.shape

        # Token embedding
        h = self.tok_emb(x)

        # Prepend meta tokens
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)  # (B, M+T, D)

        # Forward through blocks WITHOUT KV cache (training mode)
        # Note: During training, we don't use KV cache as we have full sequence
        attn_weights_list = []

        for li, block in enumerate(self.blocks):
            if return_attn:
                h, attn_w = block(h, kv_cache=None, role="owner", return_attn=True)
                attn_weights_list.append(attn_w)
            else:
                h, _ = block(h, kv_cache=None, role="owner", return_attn=False)

        # Remove meta tokens before final projection
        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:, :]  # (B, T, D)

        # Output
        h = self.norm(h)
        logits = self.head(h)  # (B, T, V)

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
        Autoregressive generation with optional KV caching and KV sharing

        Args:
            idx: Input token IDs (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
            use_kv_cache: Whether to use KV cache for efficiency

        Returns:
            Generated token IDs (B, T + max_new_tokens)
        """
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)

        if not use_kv_cache:
            # Simple generation without cache (slower but simpler)
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

        # Generation with KV cache and KV sharing
        B = idx.size(0)

        # Embed input tokens
        h = self.tok_emb(idx)

        # Prepend meta tokens
        if self.meta_tokens is not None:
            meta = self.meta_tokens.expand(B, -1, -1)
            h = torch.cat([meta, h], dim=1)

        # Prefill: process all input tokens with KV cache
        kv_caches = {}
        for li, block in enumerate(self.blocks):
            owner_id = self.owner[li]
            role = "owner" if li == owner_id else "follower"
            kv_cache = kv_caches.get(owner_id, None)

            h, new_cache = block(h, kv_cache, role, return_attn=False)

            if role == "owner":
                kv_caches[owner_id] = new_cache

        # Remove meta tokens
        if self.meta_tokens is not None:
            h = h[:, self.cfg.num_meta_tokens:, :]

        h = self.norm(h)
        logits = self.head(h)[:, -1, :]

        # Sample first new token
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

        # Autoregressive generation loop
        for _ in range(max_new_tokens - 1):
            # Embed only the new token
            h = self.tok_emb(next_token)

            # No meta tokens during incremental decoding (they're in cache)

            # Forward through blocks with cache update
            for li, block in enumerate(self.blocks):
                owner_id = self.owner[li]
                role = "owner" if li == owner_id else "follower"
                kv_cache = kv_caches.get(owner_id, None)

                h, new_cache = block(h, kv_cache, role, return_attn=False)

                if role == "owner":
                    kv_caches[owner_id] = new_cache

            h = self.norm(h)
            logits = self.head(h)[:, -1, :]

            # Sample next token
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def layer_table(self):
        import pandas as pd
        data = []
        for li in range(self.cfg.n_layers):
            is_local = li in self.swa_layers
            use_diff = self.cfg.use_differential and not is_local
            if is_local:
                attn_type = "LOCAL(SWA)"
            elif use_diff:
                attn_type = "GLOBAL(DIFF)"
            else:
                attn_type = "GLOBAL"
            data.append({
                "layer": li,
                "attn": attn_type,
                "kv_owner": self.owner[li],
                "kv_share_group": self.kv_group_id[li],
                "differential": use_diff
            })
        return pd.DataFrame(data)


# ===================== Training =====================
@dataclass
class TrainCfg:
    seq_len: int = 512
    batch_size: int = 32
    steps: int = 10000
    lr: float = 6e-4
    warmup: int = 2000
    amp: bool = True
    grad_clip: float = 1.0


def train_loop(model, train_dl, val_dl, cfg: TrainCfg, device="cuda"):
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

    # Validation
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

    print(f"\n=== Training Complete ===")
    print(f"Steps: {step}, Time: {elapsed/60:.1f}min")
    print(f"Train loss: {loss.item():.3f}, Val loss: {val_loss:.3f}, PPL: {ppl:.2f}")
    print(f"Throughput: {tps:,} tokens/s")

    return {
        "train_loss": loss.item(),
        "val_loss": val_loss,
        "ppl": ppl,
        "tps": tps
    }


def build_everything(seq_len:int=512, bs:int=32, vocab_size:int=8000):
    """Build corpus, tokenizer, and dataloaders"""
    print("Loading corpus...")
    corpus = get_corpus()

    print(f"Training tokenizer (vocab_size={vocab_size})...")
    tok = train_unigram(corpus, vocab_size=vocab_size)

    print(f"Building dataloaders (seq_len={seq_len}, bs={bs})...")
    train_dl, val_dl = build_dataloaders(tok, corpus, seq_len=seq_len, bs=bs)

    return corpus, tok, train_dl, val_dl
