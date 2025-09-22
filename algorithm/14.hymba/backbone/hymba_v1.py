import math, time, dataclasses, typing as T
import torch, torch.nn as nn, torch.nn.functional as F
import typing

# -------- Utilities --------
class RMSNorm(nn.Module):
    def __init__(self, d:int, eps:float=1e-6):
        super().__init__(); self.eps=eps; self.w=nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, d:int, mult:float=4.0, p:float=0.0):
        super().__init__()
        h=int(d*mult); self.w1=nn.Linear(d,h, bias=False); self.w2=nn.Linear(d,h,bias=False); self.w3=nn.Linear(h,d,bias=False)
        self.drop=nn.Dropout(p)
    def forward(self, x): return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))

# -------- RoPE with cache --------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim:int, base:float=10000.0, max_pos:int=131072):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # lazy cache by length
        self.register_buffer("_inv_freq", inv, persistent=False)
        self._cache_len = 0
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

    def _build_cache(self, seqlen:int, device, dtype):
        if self._cache_len >= seqlen and self._cos is not None:
            return
        t = torch.arange(seqlen, device=device, dtype=self._inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self._inv_freq)  # [T, Dh/2]
        self._cos = torch.cos(freqs).to(dtype)
        self._sin = torch.sin(freqs).to(dtype)
        self._cache_len = seqlen

    def apply(self, x: torch.Tensor, positions: torch.Tensor):
        # x: (B, H, T, Dh)
        self._build_cache(int(positions.max().item()) + 1, x.device, x.dtype)
        # ❶ 브로드캐스팅 축을 (1,1,T,…)로 두어 B,H와 충돌하지 않게
        cos = self._cos.index_select(0, positions)[None, None, :, :]  # (1,1,T,Dh/2)
        sin = self._sin.index_select(0, positions)[None, None, :, :]  # (1,1,T,Dh/2)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)
    
# -------- Attention (GQA) --------
def build_band_mask(Tc:int, T:int, window:int, device, dtype):
    # cache-friendly banded causal mask for last T rows
    i = torch.arange(Tc, device=device)
    j = torch.arange(Tc, device=device)
    causal = (j[None,:] <= i[:,None])
    band = (j[None,:] >= (i[:,None]-window+1))
    m = causal & band
    M = torch.zeros((T, Tc), device=device, dtype=dtype).masked_fill(~m[-T:], float("-inf"))
    return M

class GQA(nn.Module):
    def __init__(self, d:int, n_heads:int, n_kv_heads:int,
                 rope:RotaryEmbedding|None, use_swa:bool, swa_window:int, pdrop:float):
        super().__init__()
        assert d % max(1,n_heads) == 0 or d==0
        self.enabled = d>0 and n_heads>0
        self.H=n_heads; self.KV = max(1, n_kv_heads) if self.enabled else 1
        self.rep = (self.H // self.KV) if self.enabled else 1
        self.Dh = (d // self.H) if self.enabled else 1
        if self.enabled:
            self.q = nn.Linear(d, self.H*self.Dh, bias=False)
            self.k = nn.Linear(d, self.KV*self.Dh, bias=False)
            self.v = nn.Linear(d, self.KV*self.Dh, bias=False)
            self.o = nn.Linear(self.H*self.Dh, d, bias=False)
        self.drop = nn.Dropout(pdrop); self.rope=rope
        self.use_swa=use_swa; self.swa_window=swa_window

        # mask cache per (Tc, T) to avoid recompute
        self._mask_key = None
        self.register_buffer("_mask_cached", None, persistent=False)

    def _get_mask(self, Tc:int, T:int, device, dtype, global_mask:torch.Tensor|None):
        key = (Tc, T, device.index if device.type == 'cuda' else -1, dtype)
        if self._mask_key != key:
            if self.use_swa:
                M = build_band_mask(Tc, T, self.swa_window, device, dtype)  # (T,Tc)
            else:
                i = torch.arange(Tc, device=device); j = torch.arange(Tc, device=device)
                causal = (j[None, :] <= i[:, None])
                M = torch.zeros((T, Tc), device=device, dtype=dtype).masked_fill(~causal[-T:], float("-inf"))
            self._mask_cached = M
            self._mask_key = key

        M = self._mask_cached  # (T,Tc)

        if global_mask is None or not bool(global_mask.any()):
            return M   # 항상 2D 반환

        # ❶ 배치가 모두 동일한 글로벌 쿼리(메타 프리픽스)인 경우만 처리
        gm = global_mask[0, -T:]  # (T,)
        if bool(gm.any()):
            full = torch.zeros_like(M)
            M = torch.where(gm[:, None], full, M)  # (T,Tc)
        return M


    def forward(self, x:torch.Tensor, kv_cache:tuple[torch.Tensor,torch.Tensor]|None=None,
                global_mask:torch.Tensor|None=None):
        if not self.enabled:
            return torch.zeros_like(x), None
        B,T,C = x.shape
        q = self.q(x).view(B,T,self.H,self.Dh).transpose(1,2)      # (B,H,T,Dh)
        k = self.k(x).view(B,T,self.KV,self.Dh).transpose(1,2)     # (B,KV,T,Dh)
        v = self.v(x).view(B,T,self.KV,self.Dh).transpose(1,2)
        if kv_cache is not None and kv_cache[0] is not None and kv_cache[0].numel()>0:
            pk,pv = kv_cache; k = torch.cat([pk,k], dim=2); v = torch.cat([pv,v], dim=2)

        # Apply RoPE once
        if self.rope is not None:
            pos_q = torch.arange(k.size(2)-T, k.size(2), device=x.device)
            pos_k = torch.arange(0, k.size(2), device=x.device)
            q = self.rope.apply(q, pos_q)
            k_full = self.rope.apply(k.repeat_interleave(self.rep, dim=1), pos_k)
        else:
            k_full = k.repeat_interleave(self.rep, dim=1)

        v_full = v.repeat_interleave(self.rep, dim=1)
        Tc = k_full.size(2)
        M2d = self._get_mask(Tc, T, x.device, q.dtype, global_mask)      # (T,Tc) 보장
        q_ = q.reshape(B*self.H, T, self.Dh)
        k_ = k_full.reshape(B*self.H, Tc, self.Dh)
        v_ = v_full.reshape(B*self.H, Tc, self.Dh)
        # ❷ 여기서만 배치/헤드로 확장
        M_ = M2d.unsqueeze(0).expand(B*self.H, -1, -1)
        out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=M_, is_causal=False,
                                             dropout_p=float(self.drop.p) if self.training else 0.0)
        out = out.view(B,self.H,T,self.Dh).transpose(1,2).reshape(B,T,self.H*self.Dh)
        out = self.o(out)
        # cache for inference only (train doesn't use it)
        new_cache = (k.detach(), v.detach())
        return out, new_cache

# -------- Hybrid Block with learnable gate (robust fusion) --------
class HybridBlock(nn.Module):
    def __init__(self, d_model:int, attn_dim:int, mamba_dim:int,
                 n_heads:int, n_kv_heads:int, rope:RotaryEmbedding|None,
                 use_swa:bool, swa_window:int, ffn_mult:float, pdrop:float, fusion:str="mean"):
        super().__init__()
        self.pre = RMSNorm(d_model)
        self.attn_dim = attn_dim; self.mamba_dim=mamba_dim; self.fusion=fusion
        self.to_a = nn.Linear(d_model, attn_dim, bias=False) if attn_dim>0 else None
        self.to_m = nn.Linear(d_model, mamba_dim, bias=False) if mamba_dim>0 else None
        self.attn = GQA(attn_dim, n_heads, n_kv_heads, rope, use_swa, swa_window, pdrop) if attn_dim>0 else None
        # Mamba fallback as identity when package missing
        try:
            from mamba_ssm import Mamba as _Mamba
            self.mamba = _Mamba(d_model=mamba_dim, d_state=16, d_conv=4, expand=2) if mamba_dim>0 else None
        except Exception:
            self.mamba = nn.Identity() if mamba_dim>0 else None
        # fusion
        if fusion=="concat":
            self.mix = nn.Linear(attn_dim+mamba_dim, d_model, bias=False)
        else:
            self.gate = nn.Parameter(torch.tensor(0.5))  # learnable blend
            self.proj_a = nn.Linear(attn_dim, d_model, bias=False) if attn_dim>0 else None
            self.proj_m = nn.Linear(mamba_dim, d_model, bias=False) if mamba_dim>0 else None
        self.ffn = SwiGLU(d_model, mult=ffn_mult, p=pdrop)
        self.drop = nn.Dropout(pdrop)

    def forward(self, x, kv_cache=None, global_mask=None, training:bool=True):
        h = self.pre(x)
        outs = []
        new_cache=None
        if self.attn is not None:
            a_in = self.to_a(h)
            a, new_cache = self.attn(a_in, kv_cache=kv_cache if not training else None, global_mask=global_mask)
            outs.append(("a", a))
        if self.mamba is not None:
            m_in = self.to_m(h)
            m = self.mamba(m_in)
            outs.append(("m", m))

        if len(outs)==2:
            if hasattr(self, "mix"):  # concat
                y = self.mix(torch.cat([outs[0][1], outs[1][1]], dim=-1))
            else:
                # stable learnable gate in [0,1]
                g = torch.sigmoid(self.gate)
                y = g*self.proj_a(outs[0][1]) + (1-g)*self.proj_m(outs[1][1])
        else:
            tag, t = outs[0]
            if hasattr(self, "mix"):
                y = self.mix(t)
            else:
                y = self.proj_a(t) if tag=="a" else self.proj_m(t)

        x = x + self.drop(y)
        x = x + self.drop(self.ffn(self.pre(x)))
        return x, new_cache

# -------- Model --------

@dataclasses.dataclass
class ModelCfg:
    vocab_size:int
    d_model:int=512; n_layers:int=12; n_heads:int=8; n_kv_heads:int=4
    attn_dim:int=256; mamba_dim:int=256
    swa_layers: typing.Union[set, str, None] = "paper"  # ← 기본을 'paper'로
    swa_window:int=256
    num_meta_tokens:int=0; meta_dropout:float=0.0
    kv_share:bool=True
    fusion:str="mean"
    max_position:int=65536
    ffn_mult:float=4.0; dropout:float=0.0

class HymbaRef(nn.Module):
    def __init__(self, cfg:ModelCfg):
        super().__init__(); self.cfg=cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(dim=(cfg.attn_dim//max(1,cfg.n_heads)), max_pos=cfg.max_position) if cfg.attn_dim>0 else None
        if isinstance(cfg.swa_layers, str):
            policy = cfg.swa_layers.lower()
            if policy in ("paper", "hybrid"):
                mid = cfg.n_layers // 2
                # first/middle/last는 Global, 나머지는 SWA
                swa_layers = {i for i in range(cfg.n_layers) if i not in (0, mid, cfg.n_layers-1)}
            elif policy == "all":
                swa_layers = set(range(cfg.n_layers))
            elif policy == "none":
                swa_layers = set()
            else:
                raise ValueError(f"Unknown swa_layers policy: {cfg.swa_layers}")
        elif cfg.swa_layers is None:
            # 보수적 기본치도 논문식으로 통일
            mid = cfg.n_layers // 2
            swa_layers = {i for i in range(cfg.n_layers) if i not in (0, mid, cfg.n_layers-1)}
        else:
            swa_layers = set(cfg.swa_layers)
        self.swa_layers = swa_layers

        self.blocks = nn.ModuleList([
            HybridBlock(cfg.d_model, cfg.attn_dim, cfg.mamba_dim, cfg.n_heads, cfg.n_kv_heads, self.rope,
                        use_swa=(i in swa_layers), swa_window=cfg.swa_window,
                        ffn_mult=cfg.ffn_mult, pdrop=cfg.dropout, fusion=cfg.fusion)
            for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model); self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.num_meta = cfg.num_meta_tokens
        self.meta = nn.Parameter(torch.randn(1, cfg.num_meta_tokens, cfg.d_model)*0.02) if cfg.num_meta_tokens>0 else None
        self.meta_drop = nn.Dropout(cfg.meta_dropout)

        # owner mapping for KV share (inference only)
        self.owner = list(range(cfg.n_layers))
        if cfg.kv_share:
            for a in range(0, cfg.n_layers-1, 2): self.owner[a+1]=a

    def forward(self, input_ids:torch.Tensor, targets:torch.Tensor|None=None, task_mask:torch.Tensor|None=None):
        B,T = input_ids.shape
        x = self.tok(input_ids)
        meta_added=0
        if self.meta is not None and T>1:
            meta = self.meta.expand(B, -1, -1)
            meta = self.meta_drop(meta) if self.training else meta
            x = torch.cat([meta, x], dim=1)
            meta_added = self.num_meta

        global_mask=None
        if meta_added>0:
            gm = torch.zeros((B, x.size(1)), dtype=torch.bool, device=x.device)
            gm[:, :meta_added] = True
            global_mask = gm

        # train path: no kv cache; inference kv-share applied only in generate()
        h=x
        for li, blk in enumerate(self.blocks):
            h, _ = blk(h, kv_cache=None, global_mask=global_mask, training=self.training)
        h = self.norm(h); logits = self.head(h)

        loss=None
        if targets is not None:
            s = meta_added
            lf = logits[:, s:s+targets.size(1), :]
            loss = F.cross_entropy(lf.reshape(-1, lf.size(-1)), targets.reshape(-1))
        return {"logits":logits, "loss":loss}

    @torch.no_grad()
    def generate(self, idx:torch.Tensor, max_new_tokens:int=64, temperature:float=1.0, top_k:int=0):
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        # Warm KV once on prompt
        kv = [None]*self.cfg.n_layers
        x = self.tok(idx); B,T = x.shape[0], x.shape[1]
        meta_added=0
        if self.meta is not None and T>1:
            x = torch.cat([self.meta.expand(B,-1,-1), x],1); meta_added=self.num_meta
        global_mask=None
        if meta_added>0:
            gm = torch.zeros((B, x.size(1)), dtype=torch.bool, device=x.device); gm[:,:meta_added]=True; global_mask=gm
        h=x
        for li, blk in enumerate(self.blocks):
            owner = self.owner[li]
            h, kv_out = blk(h, kv_cache=kv[owner], global_mask=global_mask, training=False)
            if li==owner: kv[owner]=kv_out
        h = self.norm(h); logits = self.head(h)
        for _ in range(max_new_tokens):
            x_tok = idx[:, -1:]
            x = self.tok(x_tok)
            h=x
            for li, blk in enumerate(self.blocks):
                owner = self.owner[li]
                h, kv_out = blk(h, kv_cache=kv[owner], global_mask=None, training=False)
                if li==owner: kv[owner]=kv_out
            h = self.norm(h); logits = self.head(h)
            logits = logits[:, -1, :] / max(1e-5, temperature)
            if top_k>0:
                v,_=torch.topk(logits, top_k); logits[logits<v[:,[-1]]] = -float("inf")
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs,1)
            idx = torch.cat([idx, nxt],1)
        return idx

    def estimate_kv_cache_mb(self, seq_len:int, dtype=torch.float16):
        if self.cfg.attn_dim<=0 or self.cfg.n_heads<=0: return 0.0
        Dh = self.cfg.attn_dim//self.cfg.n_heads
        KV = max(1, self.cfg.n_kv_heads)
        bytes_per = torch.finfo(dtype).bits//8
        per_owner = 2 * KV * seq_len * Dh * bytes_per
        owners = len(set(self.owner))
        return per_owner * owners / (2**20)
