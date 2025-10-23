from __future__ import annotations
import math, time, typing as T, os, warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.attention import sdpa_kernel, SDPBackend

# env / warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
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
    def __init__(self, d:int, eps:float=1e-6):
        super().__init__(); self.eps=eps; self.w=nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, d:int, mult:float=4.0, p:float=0.0):
        super().__init__()
        h=int(d*mult)
        self.w1=nn.Linear(d,h,bias=False); self.w2=nn.Linear(d,h,bias=False); self.w3=nn.Linear(h,d,bias=False)
        self.drop=nn.Dropout(p)
    def forward(self, x): return self.w3(self.drop(F.silu(self.w1(x))*self.w2(x)))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim:int, base:float=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("_inv", inv, persistent=False)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
    def _build(self, L:int, device, dtype):
        if self._cos is not None and self._cos.size(0) >= L: return
        t = torch.arange(L, device=device, dtype=self._inv.dtype)
        freqs = torch.einsum("i,j->ij", t, self._inv)
        self._cos = torch.cos(freqs).to(dtype); self._sin = torch.sin(freqs).to(dtype)
    def apply(self, x:torch.Tensor, pos:torch.Tensor):
        self._build(int(pos.max().item())+1, x.device, x.dtype)
        cos = self._cos.index_select(0, pos)[None,None,:,:]
        sin = self._sin.index_select(0, pos)[None,None,:,:]
        x1, x2 = x[..., ::2], x[..., 1::2]
        o1 = x1*cos - x2*sin; o2 = x1*sin + x2*cos
        return torch.stack([o1,o2], dim=-1).flatten(-2)

def _scaled_dot_attn(q, k, v, mask_2d: torch.Tensor | None, p: float, training: bool, is_causal: bool):
    """
    통합 SDPA 백엔드 선택:
      - CUDA + (fp16/bf16) + mask 없음 + 인과적 → FLASH 선호
      - 그 외엔 EFFICIENT → MATH 자동 폴백
    """
    # 입력 성질
    on_cuda = q.is_cuda
    dtype_ok = q.dtype in (torch.float16, torch.bfloat16)
    want_flash = on_cuda and dtype_ok and (mask_2d is None) and is_causal

    if want_flash:
        # FLASH 사용 가능 시도, 불가하면 다음 백엔드로 자동 폴백
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    else:
        backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    with sdpa_kernel(backends):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask_2d,                       # 로컬/SWA는 슬라이스로 제한했으므로 보통 None
            dropout_p=p if training else 0.0,
            is_causal=is_causal and (mask_2d is None) # 마스크가 있으면 torch가 내부에서 처리
        )


class AttnLayer(nn.Module):
    """
    GQA + RoPE
    - role="owner": K/V 계산 및 캐시 갱신/반환
    - role="follower": 오너 캐시(K/V)를 그대로 사용(Q만 계산), 캐시 갱신 안 함
    - local=True → 슬라이딩윈도우 SWA (window tokens)
    """
    def __init__(self, d:int, n_heads:int, n_kv:int, local:bool=False, window:int=256, dropout:float=0.0, num_meta_tokens:int=0):
        super().__init__()
        assert d % n_heads == 0
        self.H = n_heads; self.KV = n_kv; self.Dh = d // n_heads
        self.q = nn.Linear(d, n_heads*self.Dh, bias=False)
        self.k = nn.Linear(d, n_kv*self.Dh, bias=False)
        self.v = nn.Linear(d, n_kv*self.Dh, bias=False)
        self.o = nn.Linear(n_heads*self.Dh, d, bias=False)
        self.rope = RotaryEmbedding(self.Dh)
        self.drop = nn.Dropout(dropout)
        self.local = local; self.window = window
        self.num_meta_tokens = num_meta_tokens
        self.rep = self.H // self.KV

    def _local_slice(self, k, v, Tq:int):
        # k,v: (B, H, Tk, Dh) after repeat_interleave
        if not self.local: return k, v
        Tk = k.size(2)
        w = min(self.window, Tk)
        k = k[:, :, Tk-w:Tk, :]
        v = v[:, :, Tk-w:Tk, :]
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        role: str = "owner",
        global_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """
        AttnLayer forward
        - GQA + RoPE
        - role == "owner"    : K/V를 계산(증분 concat 포함)하고 캐시 갱신/반환
        - role == "follower" : owner의 KV 캐시를 그대로 사용(Q만 계산), 캐시 갱신/반환 없음
        - self.local == True : Sliding-Window Attention (window=self.window)
        - return_attn: If True, return attention weights
        Shapes
        q: (B, H, T, Dh)
        k: owner  -> (B, KV, Tc, Dh)  (cache 포함)
            follower <- kv_cache와 동일
        k_full/v_full: KV→H 복제 후 (B, H, Tk, Dh)
        """
        B, T, C = x.shape

        # --- Query는 항상 현재 토큰들로부터 새로 계산 ---
        q = self.q(x).view(B, T, self.H, self.Dh).transpose(1, 2)  # (B,H,T,Dh)

        if role == "follower":
            # 팔로워: 오너의 K/V 캐시를 그대로 사용 (K/V 투영/concat/갱신 없음)
            assert kv_cache is not None and kv_cache[0] is not None, "Follower requires owner's KV cache"
            k_owner, v_owner = kv_cache  # (B, KV, Tc, Dh)
            # KV→Q Heads 복제
            k_full = k_owner.repeat_interleave(self.rep, dim=1)     # (B,H,Tc,Dh)
            v_full = v_owner.repeat_interleave(self.rep, dim=1)
            Tc = k_full.size(2)

            # RoPE: follower는 현재 step의 q 위치만 적용 (k는 캐시에 이미 반영되어 있다고 가정)
            if self.rope is not None:
                pos_q = torch.arange(Tc - T, Tc, device=x.device)
                q = self.rope.apply(q, pos_q)                       # (B,H,T,Dh)

            # 로컬(SWA) 윈도우 슬라이스
            k_full, v_full = self._local_slice(k_full, v_full, T)   # (B,H,Tk,Dh)
            Tk = k_full.size(2)

            # SDPA (로컬도 인과 True로 사용, 슬라이스로 윈도우 제한됨)
            if return_attn:
                # Manual computation to get attention weights
                scale = 1.0 / math.sqrt(self.Dh)
                scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B, H, T, Tk)
                if Tk > 1:
                    mask = torch.triu(torch.full((T, Tk), float('-inf'), device=x.device), diagonal=Tk - T + 1)
                    scores = scores + mask.unsqueeze(0).unsqueeze(0)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn) if self.training else attn
                out = torch.matmul(attn, v_full)  # (B, H, T, Dh)
                out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
                out = self.o(out)

                # For visualization: expand SWA attention to full-size matrix
                # follower uses owner's cache, so we need to get Tc from cache
                Tc = k_cache[0].size(2) if (k_cache is not None and k_cache[0] is not None) else Tk

                if self.local and Tk < Tc:
                    # SWA case: k_full was sliced to last Tk positions: [Tc-Tk, Tc)
                    full_attn = torch.zeros(B, self.H, T, Tc, device=attn.device, dtype=attn.dtype)
                    k_slice_start = Tc - Tk

                    for t in range(T):
                        q_abs = Tc - T + t
                        desired_k_start = max(0, q_abs - self.window + 1)
                        desired_k_end = q_abs + 1
                        actual_k_start = max(desired_k_start, k_slice_start)
                        actual_k_end = min(desired_k_end, Tc)

                        if actual_k_start < actual_k_end:
                            slice_idx_start = actual_k_start - k_slice_start
                            slice_idx_end = actual_k_end - k_slice_start
                            full_attn[:, :, t, actual_k_start:actual_k_end] = attn[:, :, t, slice_idx_start:slice_idx_end]

                    attn_to_return = full_attn
                else:
                    attn_to_return = attn

                return out, None, attn_to_return  # follower는 캐시 반환/갱신 없음
            else:
                q_ = q.reshape(B * self.H, T, self.Dh)
                k_ = k_full.reshape(B * self.H, Tk, self.Dh)
                v_ = v_full.reshape(B * self.H, Tk, self.Dh)
                out = _scaled_dot_attn(q_, k_, v_, mask_2d=None,
                                    p=float(self.drop.p), training=self.training, is_causal=True)
                out = out.view(B, self.H, T, self.Dh).transpose(1, 2).reshape(B, T, self.H * self.Dh)
                return self.o(out), None  # follower는 캐시 반환/갱신 없음

        # --- owner 경로 ---
        # 현재 토큰에서 K/V 투영
        k_new = self.k(x).view(B, T, self.KV, self.Dh).transpose(1, 2)  # (B,KV,T,Dh)
        v_new = self.v(x).view(B, T, self.KV, self.Dh).transpose(1, 2)

        # 과거 캐시가 있으면 concat (증분 디코딩)
        if kv_cache is not None and kv_cache[0] is not None and kv_cache[0].numel() > 0:
            k_prev, v_prev = kv_cache                                   # (B,KV,Tprev,Dh)
            k_cat = torch.cat([k_prev, k_new], dim=2)                   # (B,KV,Tc,Dh)
            v_cat = torch.cat([v_prev, v_new], dim=2)
        else:
            k_cat, v_cat = k_new, v_new

        Tc = k_cat.size(2)

        # RoPE: q는 현재 위치 구간, k는 0..Tc-1 전체에 적용
        if self.rope is not None:
            pos_q = torch.arange(Tc - T, Tc, device=x.device)
            pos_k = torch.arange(0, Tc, device=x.device)
            q = self.rope.apply(q, pos_q)                               # (B,H,T,Dh)
            k_cat = self.rope.apply(k_cat, pos_k)                       # (B,KV,Tc,Dh)

        # KV→Q Heads 복제
        k_full = k_cat.repeat_interleave(self.rep, dim=1)               # (B,H,Tc,Dh)
        v_full = v_cat.repeat_interleave(self.rep, dim=1)

        # SDPA
        if return_attn and self.local:
            # For visualization: compute per-query windows to show true SWA pattern
            # This is different from training where we use a shared window for efficiency
            scale = 1.0 / math.sqrt(self.Dh)
            full_attn = torch.zeros(B, self.H, T, Tc, device=x.device, dtype=k_cat.dtype)

            for t in range(T):
                # Absolute position of this query
                q_abs = Tc - T + t

                # SWA window: [max(num_meta, q_abs - window + 1), q_abs]
                # BUT: meta tokens [0:num_meta] are ALWAYS included
                window_start = max(self.num_meta_tokens, q_abs - self.window + 1)
                window_end = q_abs + 1

                # Build key/value tensors: meta tokens + window
                if self.num_meta_tokens > 0:
                    # Include meta tokens [0:num_meta] + window [window_start:window_end]
                    k_meta = k_full[:, :, :self.num_meta_tokens, :]  # (B, H, M, Dh)
                    v_meta = v_full[:, :, :self.num_meta_tokens, :]
                    k_window_regular = k_full[:, :, window_start:window_end, :]  # (B, H, win_len, Dh)
                    v_window_regular = v_full[:, :, window_start:window_end, :]

                    # Concatenate: [meta | regular_window]
                    k_window = torch.cat([k_meta, k_window_regular], dim=2)
                    v_window = torch.cat([v_meta, v_window_regular], dim=2)

                    # Attention indices in full matrix: [0:num_meta] + [window_start:window_end]
                    key_indices = list(range(self.num_meta_tokens)) + list(range(window_start, window_end))
                else:
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
            # Since full_attn has correct weights at correct positions, we can use matmul directly
            out = torch.matmul(full_attn, v_full)  # (B, H, T, Dh)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
            out = self.o(out)

            return out, (k_cat, v_cat), full_attn
        elif return_attn:
            # Global attention with return_attn (non-local case)
            scale = 1.0 / math.sqrt(self.Dh)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B, H, T, Tc)
            if Tc > 1:
                mask = torch.triu(torch.full((T, Tc), float('-inf'), device=x.device), diagonal=Tc - T + 1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn) if self.training else attn
            out = torch.matmul(attn, v_full)  # (B, H, T, Dh)
            out = out.transpose(1, 2).reshape(B, T, self.H * self.Dh)
            out = self.o(out)

            # owner만 최신 캐시(k_cat, v_cat)를 반환(원본 KV 차원 유지: (B,KV,Tc,Dh))
            return out, (k_cat, v_cat), attn
        else:
            # Training path: use efficient shared window for SWA
            k_full, v_full = self._local_slice(k_full, v_full, T)           # (B,H,Tk,Dh)
            Tk = k_full.size(2)
            q_ = q.reshape(B * self.H, T, self.Dh)
            k_ = k_full.reshape(B * self.H, Tk, self.Dh)
            v_ = v_full.reshape(B * self.H, Tk, self.Dh)
            out = _scaled_dot_attn(q_, k_, v_, mask_2d=None,
                                p=float(self.drop.p), training=self.training, is_causal=True)
            out = out.view(B, self.H, T, self.Dh).transpose(1, 2).reshape(B, T, self.H * self.Dh)
            out = self.o(out)

            # owner만 최신 캐시(k_cat, v_cat)를 반환(원본 KV 차원 유지: (B,KV,Tc,Dh))
            return out, (k_cat, v_cat)


class Block(nn.Module):
    """하나의 어텐션(+FFN) 블록. local=True면 SWA, False면 Global."""
    def __init__(self, d:int, n_heads:int, n_kv:int, local:bool, window:int, dropout:float, num_meta_tokens:int=0):
        super().__init__()
        self.pre = RMSNorm(d)
        self.attn = AttnLayer(d, n_heads, n_kv, local=local, window=window, dropout=dropout, num_meta_tokens=num_meta_tokens)
        self.post = RMSNorm(d)
        self.ffn = SwiGLU(d, mult=4.0, p=dropout)
    def forward(self, x, kv_cache=None, global_mask=None, training=True, role:str="owner", return_attn:bool=False):
        h = self.pre(x)
        attn_result = self.attn(h, kv_cache=kv_cache if not training else None, role=role, global_mask=global_mask, return_attn=return_attn)

        if return_attn:
            a, new_cache, attn_weights = attn_result
            x = x + a
            x = x + self.ffn(self.post(x))
            return x, attn_weights  # Return attention weights instead of cache for visualization
        else:
            a, new_cache = attn_result
            x = x + a
            x = x + self.ffn(self.post(x))
            return x, new_cache

# ===================== Model =====================
@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 2
    dropout: float = 0.0
    seq_len: int = 512
    # SWA 설정
    swa_layers: T.Tuple[int,...] = (1,2,3,4,5,7,8,9,10)  # 예: 0,6,11 global / 그 외 SWA
    swa_window: int = 256
    # Meta Tokens: Learnable prefix tokens that aggregate global context
    # - Prepended to input embeddings (before attention layers)
    # - Act as global memory across all layers
    # - Excluded from loss calculation
    # Recommended: 4-8 tokens for typical sequence modeling tasks
    num_meta_tokens: int = 4

class HymbaV2(nn.Module):
    def __init__(self, cfg:ModelCfg):
        super().__init__(); self.cfg=cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # ---- Meta Tokens (learnable prompts)
        self.meta = None
        if cfg.num_meta_tokens > 0:
            self.meta = nn.Parameter(torch.randn(1, cfg.num_meta_tokens, cfg.d_model) * 0.02)

        self.blocks = nn.ModuleList()
        self.swa_layers = set(cfg.swa_layers)
        for li in range(cfg.n_layers):
            is_local = (li in self.swa_layers)
            self.blocks.append(Block(cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                                     local=is_local, window=cfg.swa_window, dropout=cfg.dropout,
                                     num_meta_tokens=cfg.num_meta_tokens))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ---- KV-share 그룹/owner: 좌→우 단일패스 (SWA 전용 공유)
        self.owner = list(range(cfg.n_layers))
        self.kv_group_id = [0]*cfg.n_layers
        swa = self.swa_layers
        gid = -1; i=0; N=cfg.n_layers
        while i < N:
            if i in swa:
                j=i
                while j<N and (j in swa): j+=1
                k=i
                while k<j:
                    if k+1<j:
                        gid += 1
                        self.kv_group_id[k]=gid; self.kv_group_id[k+1]=gid
                        self.owner[k]=k; self.owner[k+1]=k
                        k+=2
                    else:
                        gid += 1
                        self.kv_group_id[k]=gid; self.owner[k]=k
                        k+=1
                i=j
            else:
                gid += 1
                self.kv_group_id[i]=gid; self.owner[i]=i
                i+=1

    # ------ Teacher-forcing forward (학습/평가용 NLL) ------
    def forward(self, input_ids:torch.LongTensor, targets:torch.LongTensor|None=None, return_attn:bool=False):
        """
        Meta tokens:
          - 입력 임베딩 앞에 [num_meta_tokens]개를 prepend.
          - loss 계산 시 meta 구간을 제외한 토큰에 대해서만 CrossEntropy.

        Args:
            return_attn: If True, returns attention weights from all attention layers
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)                         # (B,T,D)
        if self.meta is not None:
            meta = self.meta.expand(B, -1, -1)           # (B,M,D)
            x = torch.cat([meta, x], dim=1)              # (B,M+T,D)

        h = x
        attn_maps = [] if return_attn else None

        for li, blk in enumerate(self.blocks):
            if return_attn:
                # Extract attention with return_attn=True
                h, attn_weights = blk(h, kv_cache=None, global_mask=None, training=True, role="owner", return_attn=True)
                attn_maps.append(attn_weights)
            else:
                h, _ = blk(h, kv_cache=None, global_mask=None, training=True, role="owner")

        h = self.norm(h)
        logits = self.head(h)                             # (B,M+T,V)

        out = {"logits": logits}
        if return_attn:
            out["attn_weights"] = attn_maps

        if targets is not None:
            if self.meta is not None:
                M = self.meta.size(1)
                # 메타토큰 M개를 건너뛰고, 실제 토큰 위치의 logits만 사용
                # logits[:, M:M+T, :] ↔ targets[:, :T]
                logits_for_loss = logits[:, M:M+T, :]
            else:
                logits_for_loss = logits[:, :-1, :]
                targets = targets[:, 1:]
            loss = F.cross_entropy(
                logits_for_loss.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            out["loss"] = loss
        return out


    # ------ Generate helper ------
    def _owner_map_for(self, kv_share:bool):
        return self.owner if kv_share else list(range(len(self.blocks)))

    def _forward_blocks_once(self, h, owners, kv, global_mask=None):
        for li, blk in enumerate(self.blocks):
            owner = owners[li]
            role = "owner" if li == owner else "follower"
            h, kv_out = blk(h, kv_cache=kv[owner], global_mask=global_mask, training=False, role=role)
            if li == owner:
                kv[owner] = kv_out
        return h, kv

    def _forward_blocks_full_recompute(self, ids, global_mask=None):
        h = self.embed(ids)
        for blk in self.blocks:
            h,_ = blk(h, kv_cache=None, global_mask=global_mask, training=False, role="owner")
        return h

    @torch.no_grad()
    def generate(self, input_ids:torch.LongTensor, max_new_tokens:int=64,
                 temperature:float=1.0, top_k:int=0, eos_token_id:int|None=None,
                 use_kv_cache:bool=True, kv_share:bool=True):
        device = next(self.parameters()).device
        self.eval()
        ids = input_ids.to(device)

        if use_kv_cache:
            owners = self._owner_map_for(kv_share)
            kv = [None]*len(self.blocks)
            h = self.embed(ids)
            h, kv = self._forward_blocks_once(h, owners, kv, global_mask=None)
        else:
            h = self._forward_blocks_full_recompute(ids, global_mask=None)

        for _ in range(max_new_tokens):
            if use_kv_cache:
                x_step = self.embed(ids[:, -1:])
                h = x_step
                h, kv = self._forward_blocks_once(h, owners, kv, global_mask=None)
                h = self.norm(h); logits = self.head(h)[:, -1, :]
            else:
                h = self._forward_blocks_full_recompute(ids, global_mask=None)
                h = self.norm(h); logits = self.head(h)[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k < logits.size(-1):
                    topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(1, topk_idx, topk_vals)
                    logits = mask
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).all()):
                break
        return ids

    # ------ Utils ------
    def layer_table(self):
        import pandas as pd
        rows=[]
        for i,_ in enumerate(self.blocks):
            rows.append({
                "layer": i,
                "attn": "LOCAL(SWA)" if i in self.swa_layers else "GLOBAL",
                "kv_owner": self.owner[i],
                "kv_share_group": self.kv_group_id[i],
            })
        return pd.DataFrame(rows)

    def estimate_kv_cache_mb(self, seq_len:int, dtype=torch.float16):
        # owner 수 기준 추정
        Dh = self.cfg.d_model // self.cfg.n_heads
        KV = max(1, self.cfg.n_kv_heads)
        bytes_per = torch.finfo(dtype).bits // 8
        owners = len(set(self.owner))
        per_owner = 2 * KV * seq_len * Dh * bytes_per
        return round(per_owner * owners / (1024**2), 3)

# ===================== Train loop =====================
@dataclass
class TrainCfg:
    seq_len:int=512
    batch_size:int=32
    steps:int=600
    lr:float=6e-4
    warmup:int=100
    amp:bool=True
    wd:float=0.1
    grad_clip:float=1.0

def adamw_param_groups(model:nn.Module, wd:float):
    decay, no_decay = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() >= 2 and "norm" not in n.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train_loop(model:HymbaV2, train_dl, val_dl, tcfg:TrainCfg, device:str="cuda"):
    import itertools, math
    from transformers import get_cosine_schedule_with_warmup
    from torch.amp import GradScaler, autocast

    torch.manual_seed(1337)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(1337)

    model.to(device).train()
    pg = adamw_param_groups(model, wd=tcfg.wd)
    opt = torch.optim.AdamW(pg, lr=tcfg.lr, betas=(0.9,0.95), eps=1e-8,
                            fused=torch.cuda.is_available())
    sch = get_cosine_schedule_with_warmup(opt, tcfg.warmup, tcfg.steps)
    scaler = GradScaler(device="cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu",
                        enabled=tcfg.amp)

    it = itertools.cycle(train_dl)
    step=0; tok_count=0; train_nll=0.0; train_tok=0
    t0=time.time()

    while step < tcfg.steps:
        xb,yb = next(it)
        xb,yb = xb.to(device,non_blocking=True), yb.to(device,non_blocking=True)
        with autocast(device_type=("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"),
                      enabled=tcfg.amp):
            out = model(xb, targets=yb)
            loss = out["loss"]

        train_nll += float(loss.detach())*xb.numel(); train_tok += xb.numel()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if tcfg.grad_clip>0: nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
        step += 1; tok_count += xb.numel()

        if step==1 or step%50==0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"[{step:5d}] loss={loss.item():.3f} lr={lr_now:.2e}")

    elapsed = time.time()-t0
    tps = int(tok_count/max(1e-9, elapsed))
    train_loss = train_nll/max(1,train_tok)

    # val
    model.eval(); val_nll=0.0; val_tok=0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=tcfg.amp and (device=="cuda")):
        for xb,yb in val_dl:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb, targets=yb)
            val_nll += float(out["loss"].detach())*xb.numel(); val_tok += xb.numel()
    val_loss = val_nll/max(1,val_tok); ppl = math.exp(val_loss)
    return {"train_loss": float(train_loss), "val_loss": float(val_loss), "ppl": float(ppl), "tps": tps}

# ===================== Build convenience =====================
def build_everything(seq_len:int=512, bs:int=32, vocab_size:int=8000, num_meta_tokens:int=4):
    """
    Build Hymba model with dataset and tokenizer.

    Args:
        seq_len: Sequence length for training
        bs: Batch size
        vocab_size: Vocabulary size for tokenizer
        num_meta_tokens: Number of learnable meta tokens (default: 4)
                        Set to 0 to disable meta tokens
    """
    text = get_corpus("karpathy/tiny_shakespeare")
    tok = train_unigram(text, vocab_size=vocab_size)
    train_dl, val_dl = build_dataloaders(tok, text, seq_len=seq_len, bs=bs)

    cfg = ModelCfg(vocab_size=tok.vocab_size, seq_len=seq_len, num_meta_tokens=num_meta_tokens)
    model = HymbaV2(cfg)
    return model, tok, train_dl, val_dl
