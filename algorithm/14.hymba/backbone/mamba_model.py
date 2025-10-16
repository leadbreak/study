"""
Mamba (Selective State Space Model) Implementation
Based on: https://arxiv.org/abs/2312.00752

Unified interface compatible with Hymba comparison framework.
Note: This is a pure PyTorch implementation without custom CUDA kernels.
For production use, consider using the official mamba-ssm package.
"""
from __future__ import annotations
import math, time, typing as T
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ===================== Shared Components =====================
from .hymba_v2 import (
    RMSNorm,
    get_corpus,
    train_unigram,
    make_stream_dataset,
    build_dataloaders,
    adamw_param_groups
)


# ===================== SSM Core =====================
class SSM(nn.Module):
    """
    Selective State Space Model core layer.

    Implements the continuous-time SSM:
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)

    With selective parameters (Δ, B, C) that depend on input x.
    """
    def __init__(self, d_inner: int, state_size: int, device=None):
        super().__init__()
        self.d_inner = d_inner
        self.state_size = state_size
        # Device will be set when the module is moved to a device
        if device is None:
            device = 'cpu'
        self.device = device

        # Input projection to Δ, B, C
        dt_rank = max(1, math.ceil(d_inner / 16))
        self.x_proj = nn.Linear(d_inner, dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # State matrix A (initialized as stable diagonal)
        # Will be initialized on proper device when moved
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, N)

        # Feedthrough D
        self.D = nn.Parameter(torch.ones(d_inner))

        # Numerical stability constants
        self.log_eps = -7.0  # log(1e-7)
        self.exp_clamp_val = 20.0

    def discretization(self, delta, B):
        """
        Discretize continuous-time SSM parameters using ZOH (Zero-Order Hold).

        Args:
            delta: Time step (B, L, d_inner)
            B: Input matrix (B, L, state_size)

        Returns:
            delta_A: Discretized A (B, L, d_inner, state_size)
            delta_B: Discretized B (B, L, d_inner, state_size)
        """
        # A is negative for stability
        A = -torch.exp(self.A_log.float())  # (d_inner, state_size)

        # Compute ΔA with clamping for numerical stability
        log_delta_A = torch.clamp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0),
            min=self.log_eps,
            max=self.exp_clamp_val
        )
        delta_A = torch.exp(log_delta_A)  # (B, L, d_inner, state_size)

        # Compute ΔB (simplified approximation: ΔB ≈ Δ * B)
        # More accurate ZOH: ΔB = (exp(ΔA) - 1) / (ΔA) * Δ * B
        # Using simplified version for stability
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, state_size)

        return delta_A, delta_B

    def forward(self, x):
        """
        SSM forward pass with selective parameters.

        Args:
            x: Input (B, L, d_inner)

        Returns:
            y: Output (B, L, d_inner)
        """
        B, L, d_inner = x.shape

        # Compute selective parameters from input
        x_proj_out = self.x_proj(x)  # (B, L, dt_rank + 2*state_size)
        dt_inter, B_ssm, C_ssm = torch.split(
            x_proj_out,
            [self.dt_proj.in_features, self.state_size, self.state_size],
            dim=-1
        )

        # Compute time step Δ (always positive via softplus)
        dt = self.dt_proj(dt_inter)  # (B, L, d_inner)
        delta = F.softplus(dt)

        # Discretize SSM parameters
        delta_A, delta_B = self.discretization(delta, B_ssm)

        # Compute input contribution
        delta_B_u = delta_B * x.unsqueeze(-1)  # (B, L, d_inner, state_size)

        # Parallel scan using log-space for stability
        log_delta_A = torch.log(torch.clamp(delta_A, min=1e-7))
        log_R = torch.cumsum(log_delta_A, dim=1)  # (B, L, d_inner, state_size)

        # R = cumulative product of delta_A
        R = torch.exp(torch.clamp(log_R, max=self.exp_clamp_val))
        exp_neg_log_R = torch.exp(torch.clamp(-log_R, max=self.exp_clamp_val))

        # S = cumsum((delta_B * u) / R)
        S_term = delta_B_u * exp_neg_log_R
        S = torch.cumsum(S_term, dim=1)

        # Hidden states: h = R * S
        h = R * S  # (B, L, d_inner, state_size)

        # Output: y = C * h + D * x
        y = torch.einsum('bln,bldn->bld', C_ssm, h)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y


# ===================== Mamba Block =====================
class MambaBlock(nn.Module):
    """
    Full Mamba block with convolution, SSM, and gating.

    Architecture:
        x → Norm → Proj → [Conv → SiLU → SSM] ⊗ [SiLU] → Proj → + Residual
    """
    def __init__(self, d_model: int, state_size: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.state_size = state_size
        self.d_conv = d_conv
        # Device will be set when module is moved to a device
        if device is None:
            device = 'cpu'
        self.device = device

        # Input projection (expand to 2 * d_inner for split into x and z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM module
        self.ssm = SSM(self.d_inner, state_size, device=device)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Normalization
        self.norm = RMSNorm(d_model, eps=1e-5)

        # Dropout
        self.dropout_res = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)

        Returns:
            output: (B, L, d_model)
        """
        B, L, D = x.shape

        # Residual
        residual = x

        # Normalize
        x_norm = self.norm(x)

        # Project and split
        xz = self.in_proj(x_norm)  # (B, L, 2 * d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Convolution branch (causal)
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Remove extra padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv_act = F.silu(x_conv)

        # SSM branch
        y_ssm = self.ssm(x_conv_act)

        # Gating
        y_gated = y_ssm * F.silu(z)

        # Output projection
        output = self.out_proj(y_gated)

        # Residual connection with dropout
        output = residual + self.dropout_res(output)

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
        # Device will be set when the model is moved via .to(device)
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=cfg.d_model,
                state_size=cfg.state_size,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                dropout=cfg.dropout,
                device=None  # Will be set via .to(device)
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
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.w)
        if hasattr(module, 'A_log'):
            nn.init.normal_(module.A_log, mean=-1.0, std=0.5)
        if hasattr(module, 'D'):
            nn.init.ones_(module.D)

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
