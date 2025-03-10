###############################
# modern_transformer.py
###############################

import math, json, time
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from transformers import MarianTokenizer

@dataclass
class ModelArgs:
    dim: int = 512              
    n_layers: int = 6           
    n_heads: int = 8            
    n_kv_heads: int = None      
    vocab_size: int = -1        
    multiple_of: int = 4       
    ffn_dim_multiplier: float = None  
    norm_eps: float = 1e-5      
    rope_theta: float = 10000.0 
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

def precompute_freqs_cis(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings.")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    L = x.shape[1]
    return freqs_cis.view(1, L, 1, x.shape[-1] // 2)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = reshape_for_broadcast(freqs_cis, x)
    x_rotated = x_complex * freqs
    x_out = torch.view_as_real(x_rotated).reshape(x.shape)
    return x_out.type_as(x)

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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, L, _ = x.shape
        xq = self.wq(x).view(B, L, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq = xq.transpose(1, 2)  # [B, num_heads, L, head_dim]
        keys = repeat_kv(xk, self.n_rep).transpose(1, 2)
        values = repeat_kv(xv, self.n_rep).transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if L > 1:
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        out = torch.matmul(probs, values)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        base_hidden = 4 * args.dim
        hidden_dim = int(2 * base_hidden / 3)
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
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h = x + self.attention(self.attn_norm(x), freqs_cis)
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
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads,
                                              args.max_seq_len * 2,
                                              device=args.device,
                                              theta=args.rope_theta)
    def forward_train(self, tokens: torch.Tensor, start_pos: int = 0):
        B, L = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[start_pos: start_pos + L].to(h.device)
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        return self.output(h).float()
    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int = 0):
        B, L = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[start_pos: start_pos + L].to(h.device)
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        return self.output(h).float()

class LLaMA:
    """
    LLaMA 래퍼: ModernTransformer와 토크나이저를 함께 관리하며, 학습 및 생성 기능을 제공합니다.
    """
    def __init__(self, model: ModernTransformer, tokenizer, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward_train(self, tokens: torch.Tensor):
        return self.model.forward_train(tokens, start_pos=0)

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor):
        return self.model.forward_inference(tokens, start_pos=0)

    def _sample_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, -1, next_token)
        return next_token

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
        # 반복된 토큰에 대해 logits를 나누는 대신, 값을 감소시킵니다.
        for token in set(generated[0].tolist()):
            if token == self.tokenizer.eos_token_id:
                continue
            # 여기서는 단순히 해당 토큰 logit에서 penalty를 빼는 방식으로 조정합니다.
            logits[0, token] -= penalty
        return logits

    @torch.inference_mode()
    def _generate_sample(self, tokens: torch.Tensor, max_new_tokens: int,
                           temperature: float = 1.0, top_p: float = 0.95,
                           repetition_penalty: float = 1.5, min_new_tokens: int = 10) -> torch.Tensor:
        self.model.eval()
        generated = tokens.clone()
        prefix_len = tokens.size(1)
        for step in range(max_new_tokens):
            if generated.size(1) <= self.args.max_seq_len:
                context = generated
                start_pos = 0
            else:
                context = generated[:, -self.args.max_seq_len:]
                start_pos = generated.size(1) - self.args.max_seq_len
            logits = self.model.forward_inference(context, start_pos)
            next_logits = logits[:, -1, :].clone()
            # 먼저 temperature 조절
            next_logits = next_logits / temperature
            # 반복 패널티 적용 (여기서는 logit에서 값을 빼는 방식 사용)
            next_logits = self._apply_repetition_penalty(next_logits, generated, repetition_penalty)
            if top_p < 1.0:
                next_token = self._sample_top_p(next_logits, top_p)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            if step < min_new_tokens and (next_token == self.tokenizer.eos_token_id).all():
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                next_token = sorted_indices[:, 1:2]
            generated = torch.cat([generated, next_token], dim=1)
            if (generated.size(1) - prefix_len) >= min_new_tokens and (next_token == self.tokenizer.eos_token_id).all():
                break
        return generated

    @torch.inference_mode()
    def _generate_beam(self, tokens: torch.Tensor, max_new_tokens: int,
                       beam_width: int, temperature: float = 1.0) -> torch.Tensor:
        self.model.eval()
        assert tokens.size(0) == 1, "Beam search supports batch size 1 only."
        beams = [(tokens, 0.0)]
        for _ in range(max_new_tokens):
            new_beams = []
            for seq, score in beams:
                if seq.size(1) <= self.args.max_seq_len:
                    context = seq
                    start_pos = 0
                else:
                    context = seq[:, -self.args.max_seq_len:]
                    start_pos = seq.size(1) - self.args.max_seq_len
                logits = self.model.forward_inference(context, start_pos)
                next_logits = logits[:, -1, :].clone() / temperature
                log_probs = torch.log_softmax(next_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all((beam[0][0, -1].item() == self.tokenizer.eos_token_id) for beam in beams):
                break
        best_seq = beams[0][0]
        return best_seq

    def generate(self, prompt: str, max_new_tokens: int = 50, beam_width: int = 1,
                 temperature: float = 1.0, top_p: float = 0.95, repetition_penalty: float = 1.5) -> str:
        prompt = prompt.strip()
        if not prompt.endswith("=>"):
            prompt = prompt + " => "
        else:
            prompt = prompt + " "
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.args.device)
        prefix_len = prompt_tokens.size(1)
        if prefix_len > self.args.max_seq_len:
            raise ValueError("Prompt length exceeds maximum sequence length.")
        if beam_width == 1:
            generated_tokens = self._generate_sample(prompt_tokens, max_new_tokens,
                                                     temperature, top_p, repetition_penalty)
        else:
            generated_tokens = self._generate_beam(prompt_tokens, max_new_tokens, beam_width, temperature)
        target_tokens = generated_tokens[0][prefix_len:].tolist()
        if self.tokenizer.eos_token_id in target_tokens:
            target_tokens = target_tokens[:target_tokens.index(self.tokenizer.eos_token_id)]
        translation = self.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
        return translation

    @staticmethod
    def build(checkpoint_path: str, tokenizer_name: str, device: str):
        from pathlib import Path
        with open(Path(checkpoint_path).parent / "params.json", "r") as f:
            params_dict = json.load(f)
        args = ModelArgs(**params_dict)
        args.device = device
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
        args.vocab_size = tokenizer.vocab_size
        model = ModernTransformer(args).to(device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded pretrained model from {checkpoint_path}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model summary: Total trainable parameters: {total_params:,}")
        return LLaMA(model, tokenizer, args)
