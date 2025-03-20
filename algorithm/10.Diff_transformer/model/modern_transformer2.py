import json
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import MarianTokenizer

# define parameters smaller than the original LLaMA model
# to test simple training and inference

@dataclass
class ModelArgs:
    DIM = 256 # LLaMA: 4096
    FFN_DIM = DIM*4 # LLaMA: 16384
    NUM_HEADS = 8 # LLaMA: 32
    NUM_LAYERS = 4 # LLaMA: 64

    NUM_KV_HEADS = NUM_HEADS // 2 # LLaMA: 8
    VOCAB_SIZE = -1 # LLaMA: 128256 - decided by the tokenizer
    NORM_EPS = 1e-5 # LLaMA: 1e-5
    ROPE_THETA = 10000 # LLaMA: 10000

    MAX_BATCH_SIZE = 8 # depending on the GPU memory
    MAX_SEQ_LEN = 64 # depending on the DATASET
    NUM_KV_HEAD_REP = NUM_HEADS // NUM_KV_HEADS

    HEAD_DIM = DIM // NUM_HEADS
    DROPOUT = 0.1
    DEVICE = 'cpu'
    
    # EXTRA THINGS
    no_repeat_ngram_size = 3
    
# torch.set_default_dtype(torch.bfloat16)
# torch.set_default_device(ModelArgs.DEVICE)

# args = ModelArgs()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)
    
def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0, device: str = "cuda"):
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

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        hidden_dim = ffn_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        return self.w2(F.silu(self.w1(x)) * self.dropout(self.w3(x)))
    
# GQA with KV cache
class SelfAttention(nn.Module):
    def __init__(self, n_heads, n_kv_heads, n_rep, dim, dropout, batch, seq_len, device):
        super().__init__()
        self.n_heads_q = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_rep
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.attn_dropout = dropout
        
        # KV Cache
        self.cache_k = torch.zeros(batch, seq_len, n_kv_heads, self.head_dim, device=device)
        self.cache_v = torch.zeros(batch, seq_len, n_kv_heads, self.head_dim, device=device)
        
        self.norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, start_pos, freqs_cis: torch.Tensor, mask):
        B, L, _ = x.shape
        xq = self.wq(x).view(B, L, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)
        
        # apply position embedding
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        # update cache        
        self.cache_k[:B, start_pos:start_pos+L] = xk
        self.cache_v[:B, start_pos:start_pos+L] = xv
        
        # GQA 
        # Query: [B, n_heads, L, head_dim] -> [B, L, n_heads, head_dim]
        xq = xq.transpose(1, 2)
        # Key & Value: [B, n_kv_heads, L, head_dim] -> [B, n_heads, L, head_dim] -> [B, L, n_heads, head_dim]
        xk = torch.repeat_interleave(xk, repeats=self.n_rep, dim=2).transpose(1, 2)
        xv = torch.repeat_interleave(xv, repeats=self.n_rep, dim=2).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(xq, xk, xv, mask, self.attn_dropout)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out) # [B, L, D]
    
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = SelfAttention(args.NUM_HEADS, args.NUM_KV_HEADS, args.NUM_KV_HEAD_REP, args.DIM, args.DROPOUT, args.MAX_BATCH_SIZE, args.MAX_SEQ_LEN, args.DEVICE)
        self.ffn = FeedForward(args.DIM, args.FFN_DIM, args.DROPOUT)
        self.attention_norm = RMSNorm(args.DIM, args.NORM_EPS)
        self.ffn_norm = RMSNorm(args.DIM, args.NORM_EPS)
        self.res_dropout = nn.Dropout(args.DROPOUT)
    def forward(self, x: torch.Tensor, start_pos, freqs_cis: torch.Tensor, mask):
        # [B, L, D]
        h = x + self.res_dropout(self.attention(self.attention_norm(x), start_pos, freqs_cis, mask))
        o = h + self.res_dropout(self.ffn(self.ffn_norm(h)))
        return o
    
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.VOCAB_SIZE, args.DIM)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.NUM_LAYERS)])
        self.norm = RMSNorm(args.DIM, args.NORM_EPS)
        self.output = nn.Linear(args.DIM, args.VOCAB_SIZE, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.HEAD_DIM, args.MAX_SEQ_LEN, args.ROPE_THETA, args.DEVICE)
        self.device = args.DEVICE
        
    def forward_train(self, x: torch.Tensor, start_pos):
        B, L = x.shape
        h = self.tok_embeddings(x) # [B, L, D]
        freqs_cis = self.freqs_cis[start_pos:start_pos+L]
        
        mask = None
        if L > 1:
            mask = torch.full((L, L), float('-inf'), device=self.device)
            mask = torch.triu(mask, 1).to(self.device)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask) # [B, L, D]
        return self.output(self.norm(h)).float()
        
    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos):
        B, L = x.shape
        h = self.tok_embeddings(x) # [B, L, D]
        freqs_cis = self.freqs_cis[start_pos:start_pos+L]
        
        mask = None
        if L > 1:
            mask = torch.full((L, L), float('-inf'), device=self.device)
            mask = torch.triu(mask, 1).to(self.device)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask) # [B, L, D]
        return self.output(self.norm(h)).float()
    
# LLaMA 래퍼: 생성 및 학습 인터페이스 제공
class LLaMA:
    def __init__(self, model: Transformer, tokenizer, args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward_train(self, tokens: torch.Tensor):
        # tokens는 이미 decoder_input_ids임에 주의
        return self.model.forward_train(tokens, start_pos=0)

    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor):
        return self.model.forward(tokens, start_pos=0)

    # 반복 패널티: 토큰이 음수면 곱, 양수면 나눔
    def _apply_repetition_penalty(self, logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
        for token in set(generated[0].tolist()):
            if token == self.tokenizer.eos_token_id:
                continue
            if logits[0, token] < 0:
                logits[0, token] *= penalty
            else:
                logits[0, token] /= penalty
        return logits

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

    # 추가: no-repeat n-gram 체크. 만약 마지막 no_repeat_ngram_size 토큰이 모두 동일하면, 해당 토큰 logit을 -inf 처리.
    def _prevent_repeat_ngram(self, generated: torch.Tensor) -> int:
        # generated: [1, seq_len]
        n = self.args.no_repeat_ngram_size
        if generated.size(1) < n:
            return None
        last_ngram = generated[0, -n:]
        if len(set(last_ngram.tolist())) == 1:
            # 동일한 토큰 n-그램이 발생하면, 해당 토큰 id 반환 (외부에서 페널티 적용)
            return last_ngram[0].item()
        return None

    # 자기회귀 생성 (캐시 제거, 최소 생성 길이 및 no-repeat n-gram 적용)
    @torch.inference_mode()
    def _generate_sample(self, tokens: torch.Tensor, max_new_tokens: int,
                         temperature: float = 1.0, top_p: float = 0.95,
                         repetition_penalty: float = 2.0, min_new_tokens: int = 10) -> torch.Tensor:
        self.model.eval()
        generated = tokens.clone()
        prefix_len = tokens.size(1)
        for step in range(max_new_tokens):
            if generated.size(1) <= self.args.MAX_SEQ_LEN:
                context = generated
                start_pos = 0
            else:
                context = generated[:, -self.args.MAX_SEQ_LEN:]
                start_pos = generated.size(1) - self.args.MAX_SEQ_LEN
            logits = self.model.forward(context, start_pos)
            next_logits = logits[:, -1, :].clone()
            # 최소 생성 길이 미만이면 EOS 토큰 선택 방지
            if (generated.size(1) - prefix_len) < min_new_tokens:
                next_logits[:, self.tokenizer.eos_token_id] = -float('inf')
            next_logits = next_logits / temperature
            next_logits = self._apply_repetition_penalty(next_logits, generated, repetition_penalty)
            # 추가: no-repeat n-gram 검사
            repeat_token = self._prevent_repeat_ngram(generated)
            if repeat_token is not None:
                next_logits[:, repeat_token] = -float('inf')
            if top_p < 1.0:
                next_token = self._sample_top_p(next_logits, top_p)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (generated.size(1) - prefix_len) >= min_new_tokens and (next_token == self.tokenizer.eos_token_id).all():
                break
        return generated

    # Beam search 생성 (캐시 제거, 최소 생성 길이 및 no-repeat n-gram 적용)
    @torch.inference_mode()
    def _generate_beam(self, tokens: torch.Tensor, max_new_tokens: int,
                       beam_width: int, temperature: float = 1.0, min_new_tokens: int = 10) -> torch.Tensor:
        self.model.eval()
        assert tokens.size(0) == 1, "Beam search supports batch size 1 only."
        beams = [(tokens, 0.0)]
        prefix_len = tokens.size(1)
        for _ in range(max_new_tokens):
            new_beams = []
            for seq, score in beams:
                if seq.size(1) <= self.args.MAX_SEQ_LEN:
                    context = seq
                    start_pos = 0
                else:
                    context = seq[:, -self.args.MAX_SEQ_LEN:]
                    start_pos = seq.size(1) - self.args.MAX_SEQ_LEN
                    print('scout: ', start_pos)
                    raise
                logits = self.model.forward(context, start_pos=0)
                next_logits = logits[:, -1, :].clone() / temperature
                if (seq.size(1) - prefix_len) < min_new_tokens:
                    next_logits[:, self.tokenizer.eos_token_id] = -float('inf')
                log_probs = torch.log_softmax(next_logits, dim=-1)
                # 추가: no-repeat n-gram 검사
                repeat_token = self._prevent_repeat_ngram(seq)
                if repeat_token is not None:
                    log_probs[:, repeat_token] = -float('inf')
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[0, i].item()
                    new_beams.append((new_seq, new_score))
            if not new_beams:
                break
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all((beam[0][0, -1].item() == self.tokenizer.eos_token_id) for beam in beams) and \
               (beams[0][0].size(1) - prefix_len) >= min_new_tokens:
                break
        best_seq = beams[0][0]
        return best_seq

    # 외부 인터페이스: prompt를 입력받아 번역 문자열 생성
    def generate(self, prompt: str, max_new_tokens: int = 40, beam_width: int = 1,
                 temperature: float = 1.0, top_p: float = 0.95, repetition_penalty: float = 1.1,
                 min_new_tokens: int = 5) -> str:
        prompt = prompt.strip()
        if not prompt.endswith(" =>"):
            prompt += " => "
        else:
            prompt += " "
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.args.DEVICE)
        prefix_len = prompt_tokens.size(1)
        if prefix_len > self.args.MAX_SEQ_LEN:
            raise ValueError("Prompt length exceeds maximum sequence length.")
        if beam_width == 1:
            generated_tokens = self._generate_sample(prompt_tokens, max_new_tokens,
                                                     temperature, top_p, repetition_penalty, min_new_tokens)
        else:
            generated_tokens = self._generate_beam(prompt_tokens, max_new_tokens, beam_width, temperature, min_new_tokens)
        target_tokens = generated_tokens[0].tolist()
        if self.tokenizer.eos_token_id in target_tokens[prefix_len:]:
            eos_index = target_tokens[prefix_len:].index(self.tokenizer.eos_token_id)
            target_tokens = target_tokens[:prefix_len + eos_index]
        translation = self.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
        return translation

    @staticmethod
    def build(checkpoint_path: str, tokenizer_name: str, device: str):
        from pathlib import Path
        with open(Path(checkpoint_path).parent / "params.json", "r") as f:
            params_dict = json.load(f)
        args = ModelArgs(**params_dict)
        args.DEVICE = device
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
        args.VOCAB_SIZE = tokenizer.vocab_size
        model = Transformer(args).to(device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded pretrained model from {checkpoint_path}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model summary: Total trainable parameters: {total_params:,}")
        return LLaMA(model, tokenizer, args)


if __name__ == '__main__':
    params = ModelArgs()
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    params.VOCAB_SIZE = tokenizer.vocab_size
    model = Transformer(params)
    llama = LLaMA(model, tokenizer, params)
    print(llama.generate("안녕하세요 =>", max_new_tokens=30))
