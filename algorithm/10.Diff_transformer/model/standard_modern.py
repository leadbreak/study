d import pandas as pd
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import MarianTokenizer
from tqdm import tqdm
import plotly.graph_objs as go
import click

########################################
# 1. RMSNorm (LayerNorm 대체)
########################################
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
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
    def extra_repr(self):
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

########################################
# 2. RoPE 임베딩 관련 함수 (rope)
########################################
def get_rotary_emb(seq_len, head_dim, device):
    # head_dim를 전체로 사용하고, 절반으로 나누어 회전 적용
    assert head_dim % 2 == 0, "head_dim은 반드시 짝수여야 합니다."
    rotary_dim_half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim_half, device=device).float() / rotary_dim_half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, rotary_dim_half)
    cos = freqs.cos()  # (seq_len, rotary_dim_half)
    sin = freqs.sin()  # (seq_len, rotary_dim_half)
    return cos, sin

def apply_rotary_emb(x, cos, sin, interleaved=True):
    # x: (batch, n_heads, seq_len, head_dim)
    # rope는 head_dim의 앞 절반에만 적용
    head_dim = x.size(-1)
    assert head_dim % 2 == 0, "head_dim은 반드시 짝수여야 합니다."
    x1 = x[..., :head_dim//2]
    x2 = x[..., head_dim//2:]
    # cos, sin shape: (seq_len, head_dim//2) -> expand to (1,1,seq_len, head_dim//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated

########################################
# 3. Standard Transformer with RoPE & RMSNorm
########################################

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

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model ({d_model})은 n_heads ({n_heads})로 나누어 떨어져야 합니다.'
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
        # reshape and permute to (batch, n_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        # ----- RoPE 적용 (쿼리와 키에 대해) -----
        seq_len = Q.shape[2]
        cos, sin = get_rotary_emb(seq_len, self.head_dim, Q.device)
        Q = apply_rotary_emb(Q, cos, sin, interleaved=True)
        K = apply_rotary_emb(K, cos, sin, interleaved=True)
        # --------------------------------------
        attention_score = Q @ K.transpose(-1, -2) / self.scale
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -1e10)
        attention_dist = torch.softmax(attention_score, dim=-1)
        attention = attention_dist @ V
        x = attention.permute(0,2,1,3).reshape(batch_size, -1, self.d_model)
        x = self.fc_o(x)
        return x, attention_dist

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        self.self_atten = MHA(d_model, n_heads)
        self.FF = FeedForward(d_model, d_ff, drop_p)
        # RMSNorm으로 LayerNorm 대체
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(drop_p)
    def forward(self, x, enc_mask):
        x_norm = self.norm(x)  # Pre-Norm
        output, atten_enc = self.self_atten(x_norm, x_norm, x_norm, enc_mask)
        x = x + self.dropout(output)
        x_norm = self.norm(x)
        output = self.FF(x_norm)
        x = x + self.dropout(output)
        return x, atten_enc

class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        # 기존의 pos_embedding 제거: RoPE를 self-attention에서 적용함
        self.dropout = nn.Dropout(drop_p)
        self.norm = RMSNorm(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])
    def forward(self, src, mask, atten_map_save=False):
        x = self.scale * self.input_embedding(src)
        x = self.dropout(x)
        atten_encs = []
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save:
                atten_encs.append(atten_enc[0].unsqueeze(0))
        if atten_map_save:
            atten_encs = torch.cat(atten_encs, dim=0)
        x = self.norm(x)
        return x, atten_encs

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()
        self.self_atten = MHA(d_model, n_heads)
        self.cross_atten = MHA(d_model, n_heads)
        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(drop_p)
    def process_sublayer(self, x, sublayer, norm_layer, mask=None, enc_out=None):
        x_norm = norm_layer(x)
        if enc_out is not None:
            residual, atten = sublayer(x_norm, enc_out, enc_out, mask)
        else:
            residual, atten = sublayer(x_norm, x_norm, x_norm, mask)
        return x + self.dropout(residual), atten
    def forward(self, x, enc_out, dec_mask, enc_dec_mask):
        x, atten_dec = self.process_sublayer(x, self.self_atten, self.norm, dec_mask)
        x, atten_enc_dec = self.process_sublayer(x, self.cross_atten, self.norm, enc_dec_mask, enc_out)
        x, _ = self.process_sublayer(x, self.FF, self.norm)
        return x, atten_dec, atten_enc_dec

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size):
        super().__init__()
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.input_embedding = input_embedding
        # pos_embedding 제거 (RoPE 사용)
        self.dropout = nn.Dropout(drop_p)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save=False):
        x = self.scale * self.input_embedding(trg)
        x = self.dropout(x)
        atten_decs = []
        atten_enc_decs = []
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save:
                atten_decs.append(atten_dec[0].unsqueeze(0))
                atten_enc_decs.append(atten_enc_dec[0].unsqueeze(0))
        if atten_map_save:
            atten_decs = torch.cat(atten_decs, dim=0)
            atten_enc_decs = torch.cat(atten_enc_decs, dim=0)
        x = self.norm(x)
        x = self.fc_out(x)
        return x, atten_decs, atten_enc_decs

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        input_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p)
        self.decoder = Decoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p, vocab_size)
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
        trg_dec_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1], device=trg.device)) == 0
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
                torch.save({"model": model, "ep": ep, "optimizer": optimizer.state_dict(), "loss": val_loss}, loss_path)
                print(f"| Epoch {ep+1}/{EPOCH} | train loss: {train_loss:.5f} val loss: {val_loss:.5f} current_LR: {current_lr:.8f} time: {epoch_time:.2f}s => Model Saved!")
            else:
                print(f"| Epoch {ep+1}/{EPOCH} | train loss: {train_loss:.5f} val loss: {val_loss:.5f} current_LR: {current_lr:.8f} time: {epoch_time:.2f}s")
    torch.save({"loss_history": history, "EPOCH": EPOCH, "BATCH_SIZE": BATCH_SIZE}, save_history_path)
    show_history(history=history, EPOCH=EPOCH, save_path=save_model_path)

@click.command()
@click.option('--batch', default=128, help='batch size')
@click.option('--epoch', default=100, help='train epoch')
@click.option('--device', default='cuda:0', help='cuda:index')
@click.option('--model_size', default='small', help='select among [base] or [small]')
@click.option('--criterion_type', default='ce', help='select among [ce] or [lsce]')
@click.option('--label_smoothing', default=0.1, help='ratio of label smoothing')
def main(batch: int = 128, epoch: int = 100, device: str = 'cuda:0', model_size: str = 'small',
         criterion_type: str = 'ce', label_smoothing: float = 0.1):
    styled_text = click.style("Train Transformer Translator Kor-En is Started!", fg='green', bold=True)
    click.echo(styled_text)
    
    params = {}
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
    
    data = pd.read_excel('대화체.xlsx')
    custom_DS = CustomDataset(data)
    train_DS, val_DS = torch.utils.data.random_split(custom_DS, [99000, 1000])
    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
    
    vocab_size = tokenizer.vocab_size
    model = Transformer(vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p, pad_idx).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    params['scheduler'] = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)
    
    Train(model, train_DL, val_DL, criterion, optimizer, params)
    
    styled_text = click.style('Train is done!', fg='cyan', bold=True)
    click.echo(styled_text)

if __name__ == "__main__":
    main()
