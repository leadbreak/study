"""
Standard Transformer Implementation
Based on "Attention is All You Need" with modern improvements (RMSNorm, SwiGLU, RoPE)

Unified interface compatible with Hymba comparison framework.
"""
from __future__ import annotations
import math, time, typing as T
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Shared Components =====================
from .hymba_v2 import (
    RMSNorm,
    SwiGLU,
    RotaryEmbedding,
    get_corpus,
    train_unigram,
    make_stream_dataset,
    build_dataloaders,
    adamw_param_groups
)


# ===================== Multi-Head Attention =====================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q heads (GQA)"""
    B, H, L, D = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention with:
    - Grouped Query Attention (GQA)
    - RoPE (Rotary Positional Embeddings)
    - Optional KV caching for inference
    """
    def __init__(self, d: int, n_heads: int, n_kv: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.H = n_heads
        self.KV = n_kv
        self.Dh = d // n_heads
        self.rep = self.H // self.KV

        self.wq = nn.Linear(d, n_heads * self.Dh, bias=False)
        self.wk = nn.Linear(d, n_kv * self.Dh, bias=False)
        self.wv = nn.Linear(d, n_kv * self.Dh, bias=False)
        self.wo = nn.Linear(n_heads * self.Dh, d, bias=False)

        self.rope = RotaryEmbedding(self.Dh)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv_cache: T.Tuple[torch.Tensor, torch.Tensor] | None = None,
                return_attn: bool = False):
        """
        Args:
            x: (B, T, D)
            kv_cache: Optional cached (k, v) for inference
            return_attn: If True, return attention weights

        Returns:
            output: (B, T, D)
            new_cache: Updated (k, v) cache
            attn_weights: (B, H, T, Tc) if return_attn=True
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.H, self.Dh).transpose(1, 2)      # (B, H, T, Dh)
        k = self.wk(x).view(B, T, self.KV, self.Dh).transpose(1, 2)     # (B, KV, T, Dh)
        v = self.wv(x).view(B, T, self.KV, self.Dh).transpose(1, 2)     # (B, KV, T, Dh)

        # Handle KV cache for inference
        if kv_cache is not None and kv_cache[0] is not None and kv_cache[0].numel() > 0:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        Tc = k.size(2)

        # Apply RoPE
        pos_q = torch.arange(Tc - T, Tc, device=x.device)
        pos_k = torch.arange(0, Tc, device=x.device)
        q = self.rope.apply(q, pos_q)
        k = self.rope.apply(k, pos_k)

        # Repeat KV heads to match Q heads (GQA)
        k_full = repeat_kv(k, self.rep)  # (B, H, Tc, Dh)
        v_full = repeat_kv(v, self.rep)  # (B, H, Tc, Dh)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.Dh)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B, H, T, Tc)

        # Causal mask
        if Tc > 1:
            mask = torch.triu(torch.full((T, Tc), float('-inf'), device=x.device), diagonal=Tc - T + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.drop(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights_dropped, v_full)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)

        new_cache = (k.detach(), v.detach())

        if return_attn:
            return out, new_cache, attn_weights
        return out, new_cache


# ===================== Transformer Block =====================
class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-normalization"""
    def __init__(self, d: int, n_heads: int, n_kv: int, dropout: float):
        super().__init__()
        self.pre = RMSNorm(d)
        self.attn = MultiHeadAttention(d, n_heads, n_kv, dropout=dropout)
        self.post = RMSNorm(d)
        self.ffn = SwiGLU(d, mult=4.0, p=dropout)

    def forward(self, x, kv_cache=None, training=True, return_attn=False):
        h = self.pre(x)
        if return_attn:
            a, new_cache, attn_weights = self.attn(h, kv_cache=kv_cache if not training else None,
                                                    return_attn=True)
            x = x + a
            x = x + self.ffn(self.post(x))
            return x, new_cache, attn_weights
        else:
            a, new_cache = self.attn(h, kv_cache=kv_cache if not training else None,
                                     return_attn=False)
            x = x + a
            x = x + self.ffn(self.post(x))
            return x, new_cache


# ===================== Model Configuration =====================
@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 2
    dropout: float = 0.0
    seq_len: int = 512


# ===================== Main Model =====================
class StandardTransformer(nn.Module):
    """Standard Transformer with unified interface"""
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor, targets: torch.LongTensor | None = None,
                return_attn: bool = False):
        """
        Teacher-forcing forward pass

        Args:
            input_ids: (B, T)
            targets: (B, T) optional targets for loss computation
            return_attn: If True, return attention weights from all layers
        """
        x = self.embed(input_ids)  # (B, T, D)

        attn_maps = []
        for blk in self.blocks:
            if return_attn:
                x, _, attn = blk(x, kv_cache=None, training=True, return_attn=True)
                attn_maps.append(attn)
            else:
                x, _ = blk(x, kv_cache=None, training=True, return_attn=False)

        x = self.norm(x)
        logits = self.head(x)  # (B, T, V)

        out = {"logits": logits}
        if return_attn:
            out["attn_weights"] = attn_maps

        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1)
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 64,
                 temperature: float = 1.0, top_k: int = 0, eos_token_id: int | None = None,
                 use_kv_cache: bool = True):
        """Autoregressive generation with optional KV caching"""
        device = next(self.parameters()).device
        self.eval()
        ids = input_ids.to(device)

        if use_kv_cache:
            kv = [None] * len(self.blocks)
            # Prefill phase
            h = self.embed(ids)
            for li, blk in enumerate(self.blocks):
                h, kv[li] = blk(h, kv_cache=kv[li], training=False, return_attn=False)
            h = self.norm(h)
            logits = self.head(h)

            # Generation loop
            for _ in range(max_new_tokens):
                x_step = self.embed(ids[:, -1:])
                h = x_step
                for li, blk in enumerate(self.blocks):
                    h, kv[li] = blk(h, kv_cache=kv[li], training=False, return_attn=False)
                h = self.norm(h)
                logits = self.head(h)[:, -1, :]

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
        else:
            # No cache: full recomputation each step
            for _ in range(max_new_tokens):
                h = self.embed(ids)
                for blk in self.blocks:
                    h, _ = blk(h, kv_cache=None, training=False, return_attn=False)
                h = self.norm(h)
                logits = self.head(h)[:, -1, :]

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


# ===================== Training Configuration =====================
@dataclass
class TrainCfg:
    seq_len: int = 512
    batch_size: int = 32
    steps: int = 600
    lr: float = 6e-4
    warmup: int = 100
    amp: bool = True
    wd: float = 0.1
    grad_clip: float = 1.0


def train_loop(model: StandardTransformer, train_dl, val_dl, tcfg: TrainCfg, device: str = "cuda"):
    """Unified training loop compatible with Hymba comparison"""
    import itertools, math
    from transformers import get_cosine_schedule_with_warmup
    from torch.amp import GradScaler, autocast

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)

    model.to(device).train()
    pg = adamw_param_groups(model, wd=tcfg.wd)
    opt = torch.optim.AdamW(pg, lr=tcfg.lr, betas=(0.9, 0.95), eps=1e-8,
                            fused=torch.cuda.is_available())
    sch = get_cosine_schedule_with_warmup(opt, tcfg.warmup, tcfg.steps)
    scaler = GradScaler(device="cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu",
                        enabled=tcfg.amp)

    it = itertools.cycle(train_dl)
    step = 0
    tok_count = 0
    train_nll = 0.0
    train_tok = 0
    t0 = time.time()

    while step < tcfg.steps:
        xb, yb = next(it)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with autocast(device_type=("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"),
                      enabled=tcfg.amp):
            out = model(xb, targets=yb, return_attn=False)
            loss = out["loss"]

        train_nll += float(loss.detach()) * xb.numel()
        train_tok += xb.numel()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if tcfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sch.step()
        step += 1
        tok_count += xb.numel()

        if step == 1 or step % 50 == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(f"[{step:5d}] loss={loss.item():.3f} lr={lr_now:.2e}")

    elapsed = time.time() - t0
    tps = int(tok_count / max(1e-9, elapsed))
    train_loss = train_nll / max(1, train_tok)

    # Validation
    model.eval()
    val_nll = 0.0
    val_tok = 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=tcfg.amp and (device == "cuda")):
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, targets=yb, return_attn=False)
            val_nll += float(out["loss"].detach()) * xb.numel()
            val_tok += xb.numel()
    val_loss = val_nll / max(1, val_tok)
    ppl = math.exp(val_loss)
    return {"train_loss": float(train_loss), "val_loss": float(val_loss), "ppl": float(ppl), "tps": tps}


# ===================== Build convenience =====================
def build_everything(seq_len: int = 512, bs: int = 32, vocab_size: int = 8000):
    """Build model, tokenizer, and dataloaders"""
    text = get_corpus("karpathy/tiny_shakespeare")
    tok = train_unigram(text, vocab_size=vocab_size)
    train_dl, val_dl = build_dataloaders(tok, text, seq_len=seq_len, bs=bs)

    cfg = ModelCfg(vocab_size=tok.vocab_size, seq_len=seq_len)
    model = StandardTransformer(cfg)
    return model, tok, train_dl, val_dl
