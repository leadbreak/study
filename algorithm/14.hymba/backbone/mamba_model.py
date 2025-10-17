"""
Mamba (Selective State Space Model) Implementation
Based on: https://arxiv.org/abs/2312.00752

Unified interface compatible with Hymba comparison framework.
Uses official mamba-ssm package for efficient CUDA-optimized SSM operations.
"""
from __future__ import annotations
import math, time, typing as T
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import official mamba-ssm
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    print("Warning: mamba-ssm not available. Install with: pip install mamba-ssm")

# ===================== Shared Components =====================
from .hymba_v2 import (
    RMSNorm,
    get_corpus,
    train_unigram,
    make_stream_dataset,
    build_dataloaders,
    adamw_param_groups
)


# ===================== Mamba Block =====================
class MambaBlock(nn.Module):
    """
    Mamba block using official mamba-ssm package.

    Architecture:
        x → Norm → Mamba(SSM) → + Residual
    """
    def __init__(self, d_model: int, state_size: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, device=None):
        super().__init__()
        self.d_model = d_model

        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required but not installed. "
                "Install with: pip install mamba-ssm"
            )

        # Normalization
        self.norm = RMSNorm(d_model, eps=1e-5)

        # Official Mamba SSM layer
        self.mamba = MambaSSM(
            d_model=d_model,
            d_state=state_size,
            d_conv=d_conv,
            expand=expand,
        )

        # Dropout
        self.dropout_res = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)

        Returns:
            output: (B, L, d_model)
        """
        # Residual
        residual = x

        # Normalize
        x_norm = self.norm(x)

        # Mamba SSM
        y = self.mamba(x_norm)

        # Residual connection with dropout
        output = residual + self.dropout_res(y)

        return output


# ===================== Model Configuration =====================
@dataclass
class ModelCfg:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 12
    state_size: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    seq_len: int = 512


# ===================== Main Model =====================
class MambaModel(nn.Module):
    """Mamba language model with unified interface"""
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout_emb = nn.Dropout(cfg.dropout)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=cfg.d_model,
                state_size=cfg.state_size,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                dropout=cfg.dropout,
                device=None
            )
            for _ in range(cfg.n_layers)
        ])

        # Final norm and head
        self.norm_f = RMSNorm(cfg.d_model, eps=1e-5)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.w)

    def forward(self, input_ids: torch.LongTensor, targets: torch.LongTensor | None = None):
        """Teacher-forcing forward pass"""
        x = self.embedding(input_ids)  # (B, L, d_model)
        x = self.dropout_emb(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)

        out = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1)
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 64,
                 temperature: float = 1.0, top_k: int = 0, eos_token_id: int | None = None):
        """
        Autoregressive generation.
        Note: Mamba doesn't benefit from KV caching like attention models,
        but we maintain the interface for consistency.
        """
        device = next(self.parameters()).device
        self.eval()
        ids = input_ids.to(device)

        for _ in range(max_new_tokens):
            # Full forward pass (Mamba doesn't use KV cache)
            logits = self.forward(ids)["logits"][:, -1, :]

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


def train_loop(model: MambaModel, train_dl, val_dl, tcfg: TrainCfg, device: str = "cuda"):
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
            out = model(xb, targets=yb)
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
            out = model(xb, targets=yb)
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
    model = MambaModel(cfg)
    return model, tok, train_dl, val_dl
