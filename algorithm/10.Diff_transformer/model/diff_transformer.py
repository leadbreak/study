import math, json, time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from dataclasses import dataclass
from transformers import MarianTokenizer


@dataclass
class ModelArgs:
    dim: int = 512              
    n_layers: int = 6           
    n_heads: int = 8            
    groups: int = 2
    n_kv_heads: int = 4 
    vocab_size: int = -1        
    multiple_of: int = 12       
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
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(device)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    L = x.shape[1]
    return freqs_cis.view(1, L, 1, x.shape[-1] // 2)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).reshape(x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, L, nk, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(B, L, nk, n_rep, d).reshape(B, L, nk * n_rep, d)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * 4 * args.dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DiffAttention(nn.Module):
    def __init__(self, args: ModelArgs, depth: int = 0):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        # Diff Attention: project to double the head dimension for Q, K, V
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim // self.n_rep, bias=False)
        self.wv = nn.Linear(args.dim, args.dim // self.n_rep, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        
        # lambda parameter as Paper
        self.lamdba = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, L, _ = x.shape
        
        # Project and reshape with doubled head dimension
        xq = self.wq(x).view(B, L, 2 * self.n_heads_q, self.head_dim)  # [B, L, 2*n_heads, head_dim]
        xk = self.wk(x).view(B, L, 2 * self.n_kv_heads, self.head_dim) # [B, L, 2*n_kv_heads, head_dim]
        xv = self.wv(x).view(B, L, self.n_kv_heads, 2 * self.head_dim) # [B, L, n_kv_heads, 2*head_dim]
        
        # Apply rotary embeddings
        xq = apply_rotary_emb(xq, freqs_cis, device=x.device)
        xk = apply_rotary_emb(xk, freqs_cis, device=x.device)
        
        # Transpose and repeat keys/values to match query heads
        xq = xq.transpose(1, 2)  # [B, 2*n_heads, L, head_dim]
        keys = repeat_kv(xk.transpose(1, 2), self.n_rep)    # [B, 2*n_heads, L, head_dim]
        values = repeat_kv(xv.transpose(1, 2), self.n_rep)  # [B, n_heads, L, 2*head_dim]
        
        # Split Q and K into two halves along the last dimension
        d = self.head_dim
        Q1, Q2 = xq[..., :d], xq[..., d:]                   # [B, n_heads, L, head_dim]
        K1, K2 = keys[..., :d], keys[..., d:]               # [B, n_heads, L, head_dim]
        
        # Compute scaled dot-product attention for both splits
        s = 1 / math.sqrt(d)
        scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) * s # [B, n_heads, L, L]
        scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) * s # [B, n_heads, L, L]
        
        # Mask for Casual Attention
        if L > 1:
            mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
            scores1 = scores1 + mask.unsqueeze(0).unsqueeze(0)
            scores2 = scores2 + mask.unsqueeze(0).unsqueeze(0)
            
        # Compute softmax for each score and combine them with lambda
        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        attn = attn1 - self.lamdba * attn2
        
        # Compute attention output, then reshape and project
        out = torch.matmul(attn, values)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, depth: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Pass depth to DiffAttention for lambda initialization.
        self.attention = DiffAttention(args, depth)
        self.feed_forward = FeedForward(args)
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h = x + self.attention(self.attn_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))

class DiffTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        # Pass the layer index to each TransformerBlock.
        self.layers = nn.ModuleList([TransformerBlock(args, depth=i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads,
                                              args.max_seq_len * 2,
                                              device=args.device,
                                              theta=args.rope_theta)
    def forward_train(self, tokens: torch.Tensor):
        B, L = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[:L]
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        return self.output(h).float()
    
    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor):
        B, L = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs = self.freqs_cis[:L]
        for layer in self.layers:
            h = layer(h, freqs)
        h = self.norm(h)
        return self.output(h).float()


class Diff:
    """
    Diff 클래스는 학습과 추론에 모두 사용할 수 있도록 DiffTransformer와 토크나이저를 포함합니다.
    """
    def __init__(self, model: DiffTransformer, tokenizer, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward_train(self, tokens: torch.Tensor):
        return self.model.forward_train(tokens)

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor):
        return self.model.forward_inference(tokens)

    def _sample_top_p(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        logits = logits / temperature
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

    # _generate_beam은 그대로 유지 (생략)

    @torch.inference_mode()
    def _generate_sample(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float,
                           repetition_penalty: float, min_new_tokens: int = 10) -> torch.Tensor:
        self.model.eval()
        generated = tokens.clone()
        prefix_len = tokens.size(1)
        for step in range(max_new_tokens):
            context = generated if generated.size(1) <= self.args.max_seq_len else generated[:, -self.args.max_seq_len:]
            logits = self.model.forward_inference(context)
            next_logits = logits[:, -1, :].clone()
            # repetition penalty 적용
            for token in set(generated[0].tolist()):
                if token == self.tokenizer.eos_token_id:
                    continue
                next_logits[0, token] /= repetition_penalty
            if temperature > 0:
                next_token = self._sample_top_p(next_logits, temperature, top_p)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            # 만약 최소 생성 토큰 수 미만인데 EOS가 선택되면, 두 번째로 높은 토큰 선택
            if step < min_new_tokens and (next_token == self.tokenizer.eos_token_id).all():
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                next_token = sorted_indices[:, 1:2]
            generated = torch.cat([generated, next_token], dim=1)
            if (generated.size(1) - prefix_len) >= min_new_tokens and (next_token == self.tokenizer.eos_token_id).all():
                break
        return generated

    # generate 함수: teacher forcing 기반과 달리, 추론 시 입력 프롬프트("원문 =>")만 주어지므로 exposure bias가 발생함.
    # Scheduled Sampling 방식을 적용한 훈련과 free running 생성 조건 간 불일치를 보완하기 위해,
    # 생성 시에도 최소 생성 길이를 강제하고, prompt 형식을 "원문 => "로 맞춥니다.
    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50, beam_width: int = 1,
                 temperature: float = 0.7, top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
        # 입력 프롬프트 보정: "원문 =>" 형식 유지 (끝에 공백 추가)
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
            generated_tokens = self._generate_sample(prompt_tokens, max_new_tokens, temperature, top_p, repetition_penalty)
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
        model = DiffTransformer(args).to(device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded pretrained model from {checkpoint_path}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model summary: Total trainable parameters: {total_params:,}")
        return Diff(model, tokenizer, args)
