###############################
# train.py
###############################

import time, random, os
import torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd
import click
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import MarianTokenizer
from tqdm import tqdm
from modern_transformer import ModelArgs, ModernTransformer, LLaMA

# CyclicLR 기반 scheduler 래퍼
class CyclicLRScheduler:
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, mode='triangular', cycle_momentum=False):
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode,
            cycle_momentum=cycle_momentum
        )
    def step(self):
        self.scheduler.step()
    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        src = self.data.iloc[idx]['원문']
        tgt = self.data.iloc[idx]['번역문']
        return src, tgt

def collate_fn(batch, tokenizer, max_seq_len):
    prompts = []
    prefix_lens = []
    for src, tgt in batch:
        prefix = f"{src} => "
        prompt = prefix + tgt
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompts.append(prompt_ids)
        prefix_lens.append(len(prefix_ids))
    
    max_len = min(max(len(x) for x in prompts), max_seq_len)
    padded = [
        x[:max_len] + [tokenizer.pad_token_id] * (max_len - len(x[:max_len]))
        for x in prompts
    ]
    input_ids = torch.tensor(padded, dtype=torch.long)

    # Autoregressive training:
    # decoder_input_ids: 마지막 토큰을 제외한 시퀀스
    # labels: 전체 시퀀스를 오른쪽으로 한 토큰 쉬프트한 값
    decoder_input_ids = input_ids[:, :-1].contiguous()
    labels = input_ids[:, 1:].clone()

    # 프롬프트(한글 및 "=>") 영역은 loss 계산에서 제외 (-100)
    for i, prefix_len in enumerate(prefix_lens):
        if prefix_len > 1:
            labels[i, :prefix_len - 1] = -100

    return decoder_input_ids, labels, torch.tensor(prefix_lens, dtype=torch.long)

def loss_epoch_train(llama: LLaMA, dataloader, criterion, optimizer, max_seq_len, device, scheduler, scaler):
    total_loss = 0
    N = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Training", leave=False):
        decoder_input_ids, labels, prefix_lens = batch
        decoder_input_ids = decoder_input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = llama.forward_train(decoder_input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(llama.model.parameters(), llama.args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * decoder_input_ids.size(0)
    return total_loss / N

def loss_epoch_eval(llama: LLaMA, dataloader, criterion, device):
    total_loss = 0
    N = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        decoder_input_ids, labels, prefix_lens = batch
        decoder_input_ids = decoder_input_ids.to(device)
        labels = labels.to(device)
        logits = llama.forward_inference(decoder_input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item() * decoder_input_ids.size(0)
    return total_loss / N

def Train(llama: LLaMA, train_dl, val_dl, criterion, optimizer, params):
    EPOCH = params.epoch
    max_seq_len = params.max_seq_len
    device = params.device
    scheduler = params.scheduler
    history = {"train": [], "val": [], "lr": []}
    scaler = torch.amp.GradScaler('cuda')
    for ep in range(EPOCH):
        start = time.time()
        llama.model.train()
        train_loss = loss_epoch_train(llama, train_dl, criterion, optimizer, max_seq_len, device, scheduler, scaler)
        history["train"].append(train_loss)
        lr = scheduler.get_lr()
        history["lr"].append(lr)
        llama.model.eval()
        with torch.no_grad():
            val_loss = loss_epoch_eval(llama, val_dl, criterion, device)
        history["val"].append(val_loss)
        elapsed = time.time() - start
        print(f"| Epoch {ep+1}/{EPOCH} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr:.8f} | {elapsed:.2f}s")
        sample_idx = random.randint(0, len(val_dl.dataset) - 1)
        sample_src, sample_tgt = val_dl.dataset[sample_idx]
        generated_translation = llama.generate(sample_src)
        print("=== Random Sample Generation ===")
        print("원문 => 번역:", sample_src, "=>", sample_tgt)
        print("┗생성된 번역:", generated_translation)
        print("=" * 80)
    # 저장 코드 (필요 시 주석 해제)
    # torch.save(llama.model.state_dict(), params.save_model_path)
    # torch.save(history, params.save_history_path)

@click.command()
@click.option('--batch', default=128, help='배치 사이즈')
@click.option('--epoch', default=10, help='학습 에포크')
@click.option('--device', default='cuda:0', help='디바이스')
@click.option('--model_size', default='base', help="['base' 또는 'small']")
@click.option('--criterion_type', default='lsce', help="['ce' 또는 'lsce']")
@click.option('--label_smoothing', default=0.1, help='Label Smoothing 비율')
@click.option('--max_seq_len', default=64, help='최대 시퀀스 길이')
@click.option('--grad_clip', default=1.0, help='Gradient clipping 임계값')
@click.option('--base_lr', default=1e-5, type=float, help='스케줄러 Base LR')
@click.option('--max_lr', default=3e-4, type=float, help='스케줄러 Max LR')
def main(batch, epoch, device, model_size, criterion_type, label_smoothing, max_seq_len, grad_clip, base_lr, max_lr):
    click.echo(click.style("Train LLaMA (ModernTransformer 기반) 시작", fg="green", bold=True))
    params = ModelArgs()
    params.device = device
    params.max_seq_len = max_seq_len
    params.grad_clip = grad_clip
    if model_size == 'base':
        params.dim = 512
        params.n_layers = 6
        params.n_heads = 8
        warmup_steps = 4000
    else:
        params.dim = 256
        params.n_layers = 4
        params.n_heads = 8
        warmup_steps = 500
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    params.vocab_size = tokenizer.vocab_size
    df = pd.read_excel("대화체.xlsx")
    dataset = CustomDataset(df)
    train_ds, val_ds = random_split(dataset, [99000, 1000])
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
                          num_workers=16, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len),
                        num_workers=16, pin_memory=True)
    
    model = ModernTransformer(params).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"Model loaded. Total trainable parameters: {total_params:,}")
    if criterion_type == "ce":
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    params.epoch = epoch
    os.makedirs("./results", exist_ok=True)
    params.save_model_path = f"./results/moderntransformer_{criterion_type}.pt"
    params.save_history_path = f"./results/moderntransformer_history_{criterion_type}.pt"
    
    # CyclicLR 스케줄러를 사용하여 lr 조절 (step_size_up은 warmup_steps로 설정)
    params.scheduler = CyclicLRScheduler(optimizer, base_lr, max_lr, step_size_up=warmup_steps, mode='triangular', cycle_momentum=False)
    llama = LLaMA(model, tokenizer, params)
    Train(llama, train_dl, val_dl, criterion, optimizer, params)
    click.echo(click.style("학습 완료!", fg="cyan", bold=True))

if __name__ == "__main__":
    main()
