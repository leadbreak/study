import math
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import MarianTokenizer
import pandas as pd
from tqdm import tqdm
import plotly.graph_objs as go
import click

# ==================== RMSNorm 내부 구현 ====================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

# ==================== Rotary (RoPE) 관련 코드 ====================
from typing import Optional, Union
import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x1 = tl.load(X + rotary_dim_half * stride_x_headdim, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(OUT + rotary_dim_half * stride_out_headdim, o1, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(tl.float32)
        x1 = tl.load(X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))

def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        total_seqlen, nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape  # rotary_dim_half = rotary_dim/2
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"
    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    BLOCK_K = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,
            output.stride(0) if not is_varlen else 0,
            output.stride(-3),
            output.stride(-2),
            output.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None):
        out = apply_rotary(x, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=interleaved, inplace=inplace)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(do, cos, sin, seqlen_offsets=seqlen_offsets, cu_seqlens=cu_seqlens, max_seqlen=ctx.max_seqlen, interleaved=ctx.interleaved, inplace=ctx.inplace, conjugate=True)
        return dx, None, None, None, None, None, None, None

def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[int]=None):
    return ApplyRotaryEmb.apply(x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen)

# ==================== Transformer 구성 요소 (Diff Transformer 기반) ====================
# 데이터셋 클래스
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data.loc[idx, '원문'], self.data.loc[idx, '번역문']

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale=1):
        self.optimizer = optimizer
        self.step_count = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale
        self._d_model_factor = self.LR_scale * (self.d_model ** -0.5)
    def step(self):
        self.step_count += 1
        lr = self.calculate_learning_rate()
        self.optimizer.param_groups[0]['lr'] = lr
    def calculate_learning_rate(self):
        minimum_factor = min(self.step_count ** -0.5, self.step_count * self.warmup_steps ** -1.5)
        return self._d_model_factor * minimum_factor

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.linear(x)

# 간단한 인자 전달용 클래스 (diff transformer 설정)
class DummyArgs:
    decoder_kv_attention_heads = None

# get_rotary_emb 함수 (rotary_dim은 head_dim과 동일하게 사용)
def get_rotary_emb(seq_len, head_dim, device):
    rotary_dim = head_dim
    rotary_dim_half = rotary_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim_half, device=device).float() / rotary_dim_half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()  # shape: (seq_len, rotary_dim_half)
    sin = freqs.sin()  # shape: (seq_len, rotary_dim_half)
    return cos, sin

class MultiheadDiffAttn(nn.Module):
    def __init__(self, args, embed_dim, depth, num_heads):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = args.decoder_kv_attention_heads if args.decoder_kv_attention_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        # diff transformer에서는 head 차원을 절반으로 사용
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        # 내부 구현한 RMSNorm을 사용
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    def forward(self, x, rel_pos, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        # rotary 임베딩 적용 (업데이트된 apply_rotary_emb 사용)
        cos, sin = get_rotary_emb(tgt_len, self.head_dim, x.device)
        q = apply_rotary_emb(q, cos, sin, interleaved=True)
        k = apply_rotary_emb(k, cos, sin, interleaved=True)
        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(torch.zeros([tgt_len, src_len]).float().fill_(float("-inf")).type_as(attn_weights), 1 + offset)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)
        return attn

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim).reshape(bs, n_kv_heads * n_rep, slen, head_dim)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p, depth, args):
        super().__init__()
        self.self_atten = MultiheadDiffAttn(args, embed_dim=d_model, depth=depth, num_heads=n_heads)
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_p)
    def forward(self, x, enc_mask):
        x_norm = self.LN(x)
        seq_len = x_norm.size(1)
        rel_pos = get_rotary_emb(seq_len, self.self_atten.head_dim, x_norm.device)
        attn_output = self.self_atten(x_norm, rel_pos, attn_mask=enc_mask)
        x = x + self.dropout(attn_output)
        x_norm = self.LN(x)
        ff_output = self.FF(x_norm)
        x = x + self.dropout(ff_output)
        return x, None

class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, args):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(drop_p)
        self.LN = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, drop_p, depth=i, args=args) for i in range(n_layers)])
    def forward(self, src, mask, atten_map_save=False):
        pos = torch.arange(src.shape[1], device=src.device).repeat(src.shape[0], 1)
        x = self.scale * self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)
        atten_encs = []
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save:
                atten_encs.append(atten_enc)
        if atten_map_save:
            atten_encs = torch.stack(atten_encs, dim=0)
        x = self.LN(x)
        return x, atten_encs

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model ({d_model}) must be divisible by n_heads ({n_heads}).'
        self.head_dim = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attention_score = Q @ K.permute(0, 2, 3, 1) / self.scale
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -1e10)
        attention_dist = torch.softmax(attention_score, dim=-1)
        attention = attention_dist @ V
        x = attention.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        x = self.fc_o(x)
        return x, attention_dist

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p, depth, args):
        super().__init__()
        self.self_atten = MultiheadDiffAttn(args, embed_dim=d_model, depth=depth, num_heads=n_heads)
        self.cross_atten = MHA(d_model, n_heads)
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_p)
    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        x_norm = self.LN(x)
        seq_len = x_norm.size(1)
        rel_pos = get_rotary_emb(seq_len, self.self_atten.head_dim, x_norm.device)
        self_attn_output = self.self_atten(x_norm, rel_pos, attn_mask=dec_mask)
        x = x + self.dropout(self_attn_output)
        x_norm = self.LN(x)
        cross_output, atten_enc_dec = self.cross_atten(x_norm, enc_out, enc_out, enc_dec_mask)
        x = x + self.dropout(cross_output)
        x_norm = self.LN(x)
        ff_output = self.FF(x_norm)
        x = x + self.dropout(ff_output)
        return x, None, atten_enc_dec

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size, args):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(drop_p)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p, depth=i, args=args) for i in range(n_layers)])
        self.LN = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save=False):
        pos = torch.arange(trg.shape[1], device=trg.device).repeat(trg.shape[0], 1)
        x = self.scale * self.input_embedding(trg) + self.pos_embedding(pos)
        x = self.dropout(x)
        atten_decs = []
        atten_enc_decs = []
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save:
                atten_decs.append(atten_dec)
                atten_enc_decs.append(atten_enc_dec)
        if atten_map_save:
            atten_decs = torch.stack(atten_decs, dim=0)
            atten_enc_decs = torch.stack(atten_enc_decs, dim=0)
        x = self.LN(x)
        x = self.fc_out(x)
        return x, atten_decs, atten_enc_decs

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx, args):
        super().__init__()
        self.pad_idx = pad_idx
        input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, args)
        self.decoder = Decoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size, args)
        self.n_heads = n_heads
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
    def make_enc_mask(self, src):
        enc_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_mask.repeat(1, self.n_heads, src.shape[1], 1).to(src.device)
    def make_dec_mask(self, trg):
        trg_pad_mask = (trg == self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(trg.device)
        trg_dec_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1], device=trg.device))==0
        dec_mask = trg_pad_mask | trg_dec_mask
        return dec_mask
    def make_enc_dec_mask(self, src, trg):
        enc_dec_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return enc_dec_mask.repeat(1, self.n_heads, trg.shape[1], 1).to(src.device)
    def forward(self, src, trg):
        enc_mask = self.make_enc_mask(src)
        dec_mask = self.make_dec_mask(trg)
        enc_dec_mask = self.make_enc_dec_mask(src, trg)
        enc_out, atten_encs = self.encoder(src, enc_mask)
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask)
        return out, atten_encs, atten_decs, atten_enc_decs

def loss_epoch(model, DL, criterion, optimizer=None, max_len=None, DEVICE=None, tokenizer=None, scheduler=None):
    N = len(DL.dataset)
    rloss = 0
    for src_texts, trg_texts in tqdm(DL, leave=False):
        src = tokenizer(src_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids.to(DEVICE)
        trg_texts = ['</s> ' + s for s in trg_texts]
        trg = tokenizer(trg_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt').input_ids.to(DEVICE)
        y_hat = model(src, trg[:, :-1])[0]
        loss = criterion(y_hat.permute(0, 2, 1), trg[:, 1:])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        loss_b = loss.item() * src.shape[0]
        rloss += loss_b
    return rloss / N

def show_history(history, EPOCH, save_path='train_history_ls'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=history["train"], mode='lines+markers', name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=history["val"], mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis=dict(title='Loss'), showlegend=True)
    fig.write_image(save_path+"loss.png")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, EPOCH + 1)), y=history['lr'], mode='lines+markers', name='Learning Rate'))
    fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis=dict(title='Learning Rate'), showlegend=True)
    fig.write_image(save_path+"lr.png")

def Train(model, train_DL, val_DL, criterion, optimizer, params):
    BATCH_SIZE = params['batch_size']
    EPOCH = params['epoch']
    max_len = params['max_len']
    DEVICE = params['device']
    tokenizer = params['tokenizer']
    save_model_path = params['save_model_path']
    save_history_path = params['save_history_path']
    scheduler = params['scheduler']
    history = {"train": [], "val": [], "lr":[]}
    best_loss = float('inf')
    for ep in range(EPOCH):
        start_time = time.time()
        model.train()
        train_loss = loss_epoch(model, train_DL, criterion, optimizer=optimizer, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer, scheduler=scheduler)
        history["train"].append(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, val_DL, criterion, max_len=max_len, DEVICE=DEVICE, tokenizer=tokenizer)
            history["val"].append(val_loss)
            epoch_time = time.time() - start_time
            if val_loss < best_loss:
                best_loss = val_loss
                loss_path = save_model_path + '.pt'
                torch.save({"model": model, "ep": ep, "optimizer": optimizer.state_dict(), 'loss':val_loss}, loss_path)
                print(f"| Epoch {ep+1}/{EPOCH} | train loss:{train_loss:.5f} val loss:{val_loss:.5f} current_LR:{current_lr:.8f} time:{epoch_time:.2f}s => Model Saved!")
            else:
                print(f"| Epoch {ep+1}/{EPOCH} | train loss:{train_loss:.5f} val loss:{val_loss:.5f} current_LR:{current_lr:.8f} time:{epoch_time:.2f}s")
    torch.save({"loss_history": history, "EPOCH": EPOCH, "BATCH_SIZE": BATCH_SIZE}, save_history_path)
    show_history(history=history, EPOCH=EPOCH, save_path=save_model_path)

@click.command()
@click.option('--batch', default=128, help='batch size')
@click.option('--epoch', default=100, help='train epoch')
@click.option('--device', default='cuda:0', help='cuda:index')
@click.option('--model_size', default='small', help='select among [base] or [small]')
@click.option('--criterion_type', default='ce', help='select among [ce] or [lsce]')
@click.option('--label_smoothing', default=0.1, help='ratio of label smoothing')
def main(batch: int = 128, epoch: int = 100, device: str = 'cuda:0', model_size: str = 'small', criterion_type: str = 'ce', label_smoothing: float = 0.1):
    styled_text = click.style("Train Transformer Translator Kor-En is Started!", fg='green', bold=True)
    click.echo(styled_text)
    params = dict()
    params['batch_size'] = BATCH_SIZE = batch
    params['epoch'] = epoch
    params['save_model_path'] = f'./results/translator_{criterion_type}' if criterion_type=='ce' else f'./results/translator_{criterion_type}{label_smoothing}'
    params['save_history_path'] = f'./results/translator_history_{criterion_type}.pt' if criterion_type=='ce' else f'./results/translator_history_{criterion_type}{label_smoothing}.pt'
    params['device'] = DEVICE = device
    if model_size == 'base':
        params['max_len'] = max_len = 512
        d_model = 512
        n_heads = 8
        n_layers = 6
        d_ff = 2048
        drop_p = 0.1
        warmup_steps = 4000 
        LR_scale = 1
    elif model_size == 'small':
        params['max_len'] = max_len = 80
        d_model = 256
        n_heads = 8
        n_layers = 3
        d_ff = 512
        drop_p = 0.1
        warmup_steps = 1500
        LR_scale = 2
    else:
        raise ValueError("model size should be selected in ['base', 'small']")
    params['tokenizer'] = tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
    pad_idx = tokenizer.pad_token_id
    if criterion_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    elif criterion_type == 'lsce':
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=pad_idx)
    styled_text = click.style('All params are defined!', fg='cyan', bold=True)
    click.echo(styled_text)
    click.echo(params)
    args = DummyArgs()
    data = pd.read_excel('대화체.xlsx')
    custom_DS = CustomDataset(data)
    train_DS, val_DS = torch.utils.data.random_split(custom_DS, [99000, 1000])
    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
    vocab_size = tokenizer.vocab_size
    model = Transformer(vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx, args).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    params['scheduler'] = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)
    Train(model, train_DL, val_DL, criterion, optimizer, params)
    styled_text = click.style('Train is done!', fg='cyan', bold=True)
    click.echo(styled_text)

if __name__ == "__main__":
    main()
