import time, random
import torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import pandas as pd
import click
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import MarianTokenizer
from tqdm import tqdm

# updated model components (single GPU version)
from model import ModelArgs, Transformer
from generation import Llama
# 핵심 수정 포인트:
# 1. Transformer 모델의 forward 메서드에 @torch.inference_mode() 데코레이터가 있어,
#    training 시 gradient가 계산되지 않습니다.
#    → 이를 위해, training용 forward_train 메서드를 새로 정의(또는 @torch.inference_mode() 제거)하여 gradient가 흐르도록 수정합니다.
#
# 아래 코드는 LLaMA 래퍼에 forward_train 메서드를 추가하고, training 시 이를 사용하도록 업데이트한 예시입니다.

# LLaMA 래퍼에 training/inference forward 메서드 추가
class LLaMA(Llama):
    def forward_train(self, input_ids):
        return self.model.forward_train(input_ids, start_pos=0)
    
    def forward_inference(self, input_ids):
        return self.model(input_ids, start_pos=0)

# Noam 스케줄러
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
        lr = self._d_model_factor * min(self.step_count**-0.5, self.step_count * self.warmup_steps**-1.5)
        self.optimizer.param_groups[0]['lr'] = lr
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# Custom Dataset: 각 샘플은 (원문, 번역문)
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        src = self.data.loc[idx, '원문']
        tgt = self.data.loc[idx, '번역문']
        return src, tgt

# collate_fn: 각 샘플을 "{src} => {tgt}"로 결합하고, 프롬프트 prefix 길이 계산
def collate_fn(batch, tokenizer, max_seq_len):
    prompts = []
    prefix_lens = []
    for src, tgt in batch:
        prefix = f"{src} => "
        prompt = f"{prefix}{tgt}"
        prompts.append(prompt)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        prefix_lens.append(len(prefix_ids))
    encoded = tokenizer(prompts,
                        padding="longest",
                        truncation=True,
                        max_length=max_seq_len,
                        return_tensors="pt",
                        add_special_tokens=True)
    return encoded.input_ids, torch.tensor(prefix_lens)

# create_loss_mask: 프롬프트의 prefix 부분은 -100으로 마스킹
def create_loss_mask(input_ids, prefix_lens, ignore_index=-100, pad_token_id=None):
    targets = input_ids.clone()
    B, L = input_ids.shape
    for i in range(B):
        pl = prefix_lens[i]
        targets[i, :pl] = ignore_index
    if pad_token_id is not None:
        targets[targets == pad_token_id] = ignore_index
    return targets

def loss_epoch_train(llama: LLaMA, dataloader, criterion, optimizer, max_seq_len, device, scheduler, scaler):
    total_loss = 0
    N = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids, prefix_lens = batch
        input_ids = input_ids.to(device)
        targets = create_loss_mask(input_ids, prefix_lens, ignore_index=-100, pad_token_id=llama.tokenizer.pad_token_id)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            logits = llama.forward_train(input_ids)
            loss = criterion(logits.permute(0, 2, 1), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / N

def loss_epoch_eval_with_details(llama: LLaMA, dataloader, criterion, device):
    total_loss = 0
    N = len(dataloader.dataset)
    debug_printed = False
    # 디코딩 하이퍼파라미터
    max_new_tokens = 50
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.1
    min_new_tokens = 10

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids, prefix_lens = batch
        input_ids = input_ids.to(device)
        targets = create_loss_mask(input_ids, prefix_lens, ignore_index=-100,
                                   pad_token_id=llama.tokenizer.pad_token_id)
        logits = llama.forward_inference(input_ids)
        loss = criterion(logits.permute(0, 2, 1), targets)
        total_loss += loss.item() * input_ids.size(0)
        
        if not debug_printed:
            pred_ids = torch.argmax(logits, dim=-1)
            print("\n=== Evaluation Sample Details ===")
            print(f"Batch Loss: {loss.item():.5f}")
            for i in range(min(3, input_ids.size(0))):
                decoded_input = llama.tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True)
                pred_ids_slice = pred_ids[i][prefix_lens[i]:].tolist()
                if llama.tokenizer.eos_token_id in pred_ids_slice:
                    pred_ids_slice = pred_ids_slice[:pred_ids_slice.index(llama.tokenizer.eos_token_id)]
                decoded_pred = llama.tokenizer.decode(pred_ids_slice, skip_special_tokens=True)
                target_ids = [token if token != -100 else llama.tokenizer.pad_token_id
                              for token in targets[i].tolist()]
                if llama.tokenizer.eos_token_id in target_ids:
                    target_ids = target_ids[:target_ids.index(llama.tokenizer.eos_token_id)]
                decoded_target = llama.tokenizer.decode(target_ids, skip_special_tokens=True)
                print(f"\nSample {i+1}:")
                print("Input    :", decoded_input)
                print("Target   :", decoded_target)
                print("Predicted:", decoded_pred)
                
                # 개선된 autoregressive 생성: forward_inference + sampling, repetition penalty, top-p, min 길이 적용
                prompt_ids = input_ids[i][:prefix_lens[i]].unsqueeze(0)
                generated_ids = prompt_ids.clone()
                prefix_len = prompt_ids.size(1)
                for step in range(max_new_tokens):
                    context = generated_ids if generated_ids.size(1) <= llama.model.params.max_seq_len else generated_ids[:, -llama.model.params.max_seq_len:]
                    logits_context = llama.forward_inference(context)
                    next_logits = logits_context[:, -1, :].clone()
                    for token in set(generated_ids[0].tolist()):
                        if token == llama.tokenizer.eos_token_id:
                            continue
                        next_logits[0, token] /= repetition_penalty
                    next_logits = next_logits / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    sorted_probs[sorted_indices_to_remove] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(sorted_probs, num_samples=1)
                    if step < min_new_tokens and next_token.item() == llama.tokenizer.eos_token_id:
                        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                        next_token = sorted_indices[:, 1:2]
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    if (generated_ids.size(1) - prefix_len) >= min_new_tokens and next_token.item() == llama.tokenizer.eos_token_id:
                        break
                decoded_generated = llama.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
                print("Generated:", decoded_generated[prefix_len:])
                print("-" * 40)
            debug_printed = True
    return total_loss / N

def Train(llama: LLaMA, train_dl, val_dl, val_dataset, criterion, optimizer, params):
    EPOCH = params.epoch
    max_seq_len = params.max_seq_len
    device = params.device
    scheduler = params.scheduler
    history = {"train": [], "val": [], "lr": []}
    best_loss = float("inf")
    scaler = torch.amp.GradScaler()
    for ep in range(EPOCH):
        start = time.time()
        llama.model.train()
        train_loss = loss_epoch_train(llama, train_dl, criterion, optimizer, max_seq_len, device, scheduler, scaler)
        history["train"].append(train_loss)
        lr = scheduler.get_lr()
        history["lr"].append(lr)
        llama.model.eval()
        with torch.no_grad():
            val_loss = loss_epoch_eval_with_details(llama, val_dl, criterion, device)
        history["val"].append(val_loss)
        elapsed = time.time() - start
        print(f"| Epoch {ep+1}/{EPOCH} | Train: {train_loss:.5f} | Val: {val_loss:.5f} | LR: {lr:.8f} | {elapsed:.2f}s")
        if val_loss < best_loss and ep >= EPOCH * 0.5:
            best_loss = val_loss
            torch.save(llama.model.state_dict(), params.save_model_path)
            print("=> Model Saved!")
    torch.save(history, params.save_history_path)

@click.command()
@click.option('--batch', default=128, help='배치 사이즈')
@click.option('--epoch', default=10, help='학습 에포크')
@click.option('--device', default='cuda:0', help='디바이스')
@click.option('--model_size', default='base', help="['base' 또는 'small']")
@click.option('--criterion_type', default='ce', help="['ce' 또는 'lsce']")
@click.option('--label_smoothing', default=0.1, help='Label Smoothing 비율')
@click.option('--max_seq_len', default=512, help='최대 시퀀스 길이')
def main(batch, epoch, device, model_size, criterion_type, label_smoothing, max_seq_len):
    click.echo(click.style("Train LLaMA (Updated Transformer 기반) 시작", fg="green", bold=True))
    params = ModelArgs()
    params.device = device
    params.max_seq_len = max_seq_len
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
        params.max_seq_len = max_seq_len
        warmup_steps = 1500
        LR_scale = 2
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    params.vocab_size = tokenizer.vocab_size
    params.tokenizer = tokenizer
    df = pd.read_excel("대화체.xlsx")
    dataset = CustomDataset(df)
    train_ds, val_ds = random_split(dataset, [99000, 1000])
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len))
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_seq_len))
    model = Transformer(params).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"Model loaded. Total trainable parameters: {total_params:,}")
    if criterion_type == "ce":
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    params.epoch = epoch
    params.save_model_path = f"./results/transformer_{criterion_type}.pt"
    params.save_history_path = f"./results/transformer_history_{criterion_type}.pt"
    params.scheduler = NoamScheduler(optimizer, d_model=params.dim, warmup_steps=warmup_steps, LR_scale=LR_scale)
    llama = LLaMA(model, tokenizer)
    Train(llama, train_dl, val_dl, val_ds, criterion, optimizer, params)
    click.echo(click.style("학습 완료!", fg="cyan", bold=True))

if __name__ == "__main__":
    main()

