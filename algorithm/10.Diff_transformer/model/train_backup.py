# training.py
import math
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import click
from dataclasses import dataclass
from transformers import MarianTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

########################################
# Model 정의 (ModernTransformer)
########################################

@dataclass
class ModelArgs:
    dim: int = 512               # 예: 기본 512
    n_layers: int = 6            # 예: 기본 6
    n_heads: int = 8             # 예: 기본 8
    n_kv_heads: int = None       # 없으면 n_heads로 사용
    vocab_size: int = -1
    multiple_of: int = 12
    ffn_dim_multiplier: float = None
    norm_eps: float = 1e-5
    max_batch_size: int = 1024
    max_seq_len: int = 512
    device: str = "cpu"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    theta_numer = torch.arange(0, head_dim, 2).float()
    theta_vals = 1.0 / (theta ** (theta_numer / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta_vals).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    return x_out.reshape(x.shape).type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, L, nk, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(B, L, nk, n_rep, d).reshape(B, L, nk * n_rep, d)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 캐시 (기울기 추적 차단을 위해 detach() 적용)
        self.register_buffer('cache_k', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.register_buffer('cache_v', torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, apply_causal_mask: bool = True):
        B, L, _ = x.shape
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        xq = xq.view(B, L, self.n_heads_q, self.head_dim)
        xk = xk.view(B, L, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, L, self.n_kv_heads, self.head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache_k[:B, start_pos:start_pos+L] = xk.detach()
        self.cache_v[:B, start_pos:start_pos+L] = xv.detach()

        keys = self.cache_k[:B, :start_pos+L]
        values = self.cache_v[:B, :start_pos+L]
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)  # (B, n_heads, L, head_dim)
        keys = keys.transpose(1, 2)  # (B, n_heads, total_L, head_dim)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if apply_causal_mask:
            # 현재 글로벌 위치: query i -> start_pos + i
            total_L = start_pos + L
            q_idx = torch.arange(start_pos, start_pos+L, device=x.device).unsqueeze(1)
            k_idx = torch.arange(0, total_L, device=x.device).unsqueeze(0)
            mask = k_idx > q_idx  # (L, total_L)
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-inf'))
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        out = torch.matmul(probs, values)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward_train(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.attn_norm(x), start_pos, freqs_complex, apply_causal_mask=True)
        return h + self.feed_forward(self.ffn_norm(h))
    def forward_inference(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.attn_norm(x), start_pos, freqs_complex, apply_causal_mask=False)
        return h + self.feed_forward(self.ffn_norm(h))

class ModernTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_len * 2, device=args.device)
    # 학습 시 forward
    def forward_train(self, tokens: torch.Tensor, start_pos: int = 0):
        B, L = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_complex[start_pos:start_pos+L]
        for layer in self.layers:
            h = layer.forward_train(h, start_pos, freqs)
        h = self.norm(h)
        return self.output(h).float()
    # 추론 시 forward (inference mode 외부 사용 권장)
    def forward_inference(self, tokens: torch.Tensor, start_pos: int = 0):
        B, L = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_complex[start_pos:start_pos+L]
        for layer in self.layers:
            h = layer.forward_inference(h, start_pos, freqs)
        h = self.norm(h)
        return self.output(h).float()

########################################
# 학습 관련 유틸리티
########################################

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # '원문' 컬럼만 사용 (causal LM)
        return self.data.loc[idx, '원문']

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale=1):
        self.optimizer = optimizer
        self.step_count = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale
        self.factor = self.LR_scale * (self.d_model ** -0.5)
    def step(self):
        self.step_count += 1
        lr = self.factor * min(self.step_count**-0.5, self.step_count * self.warmup_steps**-1.5)
        self.optimizer.param_groups[0]['lr'] = lr

def loss_epoch(model, dataloader, criterion, optimizer=None, max_seq_len=None, device=None, tokenizer=None, scheduler=None):
    total_loss = 0
    N = len(dataloader.dataset)
    for texts in tqdm(dataloader, desc="Training", leave=False):
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt").input_ids.to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model.forward_train(inputs)
        loss = criterion(logits.permute(0, 2, 1), targets)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss += loss.item() * tokens.size(0)
    return total_loss / N

def Train(model, train_dl, val_dl, criterion, optimizer, params):
    EPOCH = params.epoch
    max_seq_len = params.max_seq_len
    device = params.device
    tokenizer = params.tokenizer
    scheduler = params.scheduler
    history = {"train": [], "val": [], "lr": []}
    best_loss = float("inf")
    for ep in range(EPOCH):
        start = time.time()
        model.train()
        train_loss = loss_epoch(model, train_dl, criterion, optimizer, max_seq_len, device, tokenizer, scheduler)
        history["train"].append(train_loss)
        lr = optimizer.param_groups[0]['lr']
        history["lr"].append(lr)
        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, val_dl, criterion, None, max_seq_len, device, tokenizer)
        history["val"].append(val_loss)
        elapsed = time.time() - start
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), params.save_model_path)
            print(f"| Epoch {ep+1}/{EPOCH} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr:.8f} | {elapsed:.2f}s => Model Saved!")
        else:
            print(f"| Epoch {ep+1}/{EPOCH} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr:.8f} | {elapsed:.2f}s")
    # 저장 및 요약 출력
    torch.save(history, params.save_history_path)

########################################
# Main (학습)
########################################

@click.command()
@click.option('--batch', default=128, help='배치 사이즈')
@click.option('--epoch', default=10, help='학습 에포크')
@click.option('--device', default='cuda:0', help='디바이스')
@click.option('--model_size', default='base', help="['base' 또는 'small']")
@click.option('--criterion_type', default='ce', help="['ce' 또는 'lsce']")
@click.option('--label_smoothing', default=0.1, help='Label Smoothing 비율')
def main(batch, epoch, device, model_size, criterion_type, label_smoothing):
    click.echo(click.style("Train ModernTransformer (최종 LLaMA) 시작", fg="green", bold=True))
    params = ModelArgs()
    params.device = device
    params.max_seq_len = 512
    if model_size == 'base':
        params.dim = 512
        params.n_layers = 6
        params.n_heads = 8
        warmup_steps = 4000
        LR_scale = 1
    else:
        params.dim = 256
        params.n_layers = 3
        params.n_heads = 8
        params.max_seq_len = 80
        warmup_steps = 1500
        LR_scale = 2
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    params.vocab_size = tokenizer.vocab_size
    params.tokenizer = tokenizer
    df = pd.read_excel("대화체.xlsx")
    dataset = CustomDataset(df)
    train_ds, val_ds = random_split(dataset, [99000, 1000])
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    model = ModernTransformer(params).to(device)
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"Model loaded. Total parameters: {total_params:,}")
    if criterion_type == "ce":
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    params.epoch = epoch
    params.batch_size = batch
    params.save_model_path = f"./results/moderntransformer_{criterion_type}.pt"
    params.save_history_path = f"./results/moderntransformer_history_{criterion_type}.pt"
    params.scheduler = NoamScheduler(optimizer, d_model=params.dim, warmup_steps=warmup_steps, LR_scale=LR_scale)
    Train(model, train_dl, val_dl, criterion, optimizer, params)
    click.echo(click.style("학습 완료!", fg="cyan", bold=True))

if __name__ == "__main__":
    main()
