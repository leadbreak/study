# -*- coding: utf-8 -*-
"""
PyTorch Vision Transformer (ViT) 실험을 위한 CLI 애플리케이션.

이 스크립트는 'click' 라이브러리를 사용하여 두 가지 주요 시나리오를 실행합니다.
- 시나리오 A: 표준적인 사전학습 및 파인튜닝 방식
- 시나리오 B: PFN (Prior-data Fitted Network) 기반의 퓨샷(Few-shot) 메타러닝 방식

각 시나리오는 하위 명령어로 분리되어 있으며, 다양한 하이퍼파라미터를 CLI 옵션으로 제어할 수 있습니다.

[실행 예시]
# 시나리오 A 실행 (기본값 사용, 장치 자동 선택)
python your_script_name.py scenario-a

# 시나리오 A를 특정 GPU(cuda:1)에서 실행
python your_script_name.py --device cuda:1 scenario-a --pretrain-epochs 20

# 시나리오 B 실행 (커스텀 파라미터 사용)
python your_script_name.py scenario-b --epochs 1500 --n-way 5 --k-shot 10 --batch-size 32
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.transforms import functional as F_vision
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import random
import time
import logging
import os
import click
from einops.layers.torch import Rearrange

# PyTorch 2.0 이상에서 사용 가능한 컴파일 기능 활성화 여부
# 성능을 크게 향상시킬 수 있지만, 일부 환경에서는 호환성 문제가 발생할 수 있습니다.
USE_TORCH_COMPILE = hasattr(torch, 'compile')

# --- 유틸리티 함수 ---

def setup_logging(log_path: str):
    """로깅 설정을 초기화합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int):
    """재현성을 위해 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 모델 아키텍처 ---

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(emb_size, emb_size * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(emb_size * 2, emb_size))
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(emb_size), nn.LayerNorm(emb_size), nn.Dropout(dropout)
    def forward(self, src):
        x = self.norm1(src + self.dropout(self.mha(src, src, src)[0]))
        return self.norm2(x + self.dropout(self.ffn(x)))

class VisionTransformerEncoder(nn.Module):
    def __init__(self, emb_size=512, num_heads=8, patch_size=4, num_layers=6, dropout=0.1):
        super().__init__()
        image_size, patch_dim = 32, 3 * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patching_and_flatten = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, emb_size), nn.LayerNorm(emb_size),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(emb_size, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_size)
    def forward(self, x):
        b = x.shape[0]
        x = self.patching_and_flatten(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embedding
        for layer in self.transformer_layers: x = layer(x)
        return self.norm(x[:, 0])

# --- 시나리오 A: 표준 ViT 모델 및 학습/평가 함수 ---

class StandardViT(nn.Module):
    def __init__(self, num_classes, emb_size=512, num_layers=6):
        super().__init__()
        self.backbone = VisionTransformerEncoder(emb_size=emb_size, num_layers=num_layers)
        self.classifier = nn.Linear(emb_size, num_classes)
    def forward(self, x): return self.classifier(self.backbone(x))

def train_standard(model, train_loader, eval_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Standard Train)")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_time = time.time() - epoch_start_time
        val_acc = evaluate_standard(model, eval_loader, device)
        logging.info(f"Epoch {epoch+1}/{epochs} -> Val Accuracy: {val_acc:.2f}%, Time: {epoch_time:.2f}s")

def evaluate_standard(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0

# --- 시나리오 B: PFN 모델 및 학습/평가 함수 ---

class GaussianNoise(nn.Module):
    def __init__(self, stddev=1e-3):
        super().__init__()
        self.stddev = stddev
    def forward(self, x):
        if self.training and self.stddev != 0:
            random_multiplier = 3 * (torch.rand(1, device=x.device) * 2 - 1)
            noise = torch.randn_like(x) * self.stddev * random_multiplier
            return x + noise
        return x

class PosteriorInferenceViT(nn.Module):
    def __init__(self, n_way, emb_size=512, num_heads=8, num_backbone_layers=6, num_inference_layers=4, noise_stddev=3e-3, add_noise=True):
        super().__init__()
        self.n_way = n_way
        self.backbone = VisionTransformerEncoder(emb_size=emb_size, num_heads=num_heads, num_layers=num_backbone_layers)
        self.noise_injector = GaussianNoise(stddev=noise_stddev) if add_noise else nn.Identity()
        self.label_embedding = nn.Embedding(n_way + 1, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, batch_first=True, norm_first=True)
        self.inference_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_inference_layers)
        self.classifier = nn.Linear(emb_size, n_way)
    def forward(self, support_images, support_labels, query_images):
        B, num_support, C, H, W = support_images.shape
        num_query = query_images.shape[1]
        all_images = torch.cat([support_images.view(-1, C, H, W), query_images.view(-1, C, H, W)])
        all_features = self.noise_injector(self.backbone(all_images))
        support_features, query_features = all_features[:B * num_support].view(B, num_support, -1), all_features[B * num_support:].view(B, num_query, -1)
        support_features += self.label_embedding(support_labels)
        query_features += self.label_embedding(torch.full((B, num_query), self.n_way, device=support_images.device).long())
        inferred_query_features = self.inference_transformer(torch.cat([support_features, query_features], dim=1))[:, num_support:]
        return self.classifier(inferred_query_features)

def train_pfn(model, dataloader, optimizer, device, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (PFN Meta-train)")
        total_acc_epoch = 0
        for s_img, s_lbl, q_img, q_lbl in pbar:
            s_img, s_lbl, q_img, q_lbl = map(lambda x: x.to(device), [s_img, s_lbl, q_img, q_lbl])
            optimizer.zero_grad()
            logits = model(s_img, s_lbl, q_img)
            loss = criterion(logits.view(-1, model.n_way), q_lbl.view(-1))
            loss.backward(); optimizer.step()
            acc = (logits.argmax(dim=-1) == q_lbl).float().mean()
            total_acc_epoch += acc.item()
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_acc = 100 * total_acc_epoch / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{epochs} -> Meta-Train Accuracy: {avg_epoch_acc:.2f}%, Time: {epoch_time:.2f}s")

def evaluate_pfn(model, dataloader, device):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for s_img, s_lbl, q_img, q_lbl in tqdm(dataloader, desc="PFN Meta-evaluation"):
            s_img, s_lbl, q_img, q_lbl = map(lambda x: x.to(device), [s_img, s_lbl, q_img, q_lbl])
            total_acc += (model(s_img, s_lbl, q_img).argmax(dim=-1) == q_lbl).float().mean().item()
    return 100 * total_acc / len(dataloader) if len(dataloader) > 0 else 0

# --- 데이터셋 클래스 ---

class Cifar100EpisodeDataset(Dataset):
    def __init__(self, dataset, n_way, k_shot, q_queries, num_episodes):
        self.dataset, self.n_way, self.k_shot, self.q_queries = dataset, n_way, k_shot, q_queries
        self.num_episodes = num_episodes
        self.targets_np = np.array(dataset.dataset.targets)[dataset.indices] if isinstance(dataset, Subset) else np.array(dataset.targets)
        self.label_to_indices = {label: np.where(self.targets_np == label)[0] for label in np.unique(self.targets_np)}
    def __len__(self): return self.num_episodes
    def __getitem__(self, index):
        classes = random.sample(list(self.label_to_indices.keys()), self.n_way)
        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
        for i, cls in enumerate(classes):
            indices = random.sample(list(self.label_to_indices[cls]), self.k_shot + self.q_queries)
            s_indices, q_indices = indices[:self.k_shot], indices[self.k_shot:]
            s_imgs.extend([self.dataset[j][0] for j in s_indices]); s_lbls.extend([i] * self.k_shot)
            q_imgs.extend([self.dataset[j][0] for j in q_indices]); q_lbls.extend([i] * self.q_queries)
        return (torch.stack(s_imgs), torch.LongTensor(s_lbls), torch.stack(q_imgs), torch.LongTensor(q_lbls))

class RemappedDataset(Dataset):
    def __init__(self, original_dataset, indices, transform, label_offset):
        self.data = [original_dataset.data[i] for i in indices]
        self.targets = [original_dataset.targets[i] - label_offset for i in indices]
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.transform(F_vision.to_pil_image(self.data[idx])), self.targets[idx]

# --- CLI 정의 ---

@click.group()
@click.option('--device', default=None, help="실행할 장치 (e.g., 'cuda:0', 'cpu'). 지정하지 않으면 자동으로 선택합니다.")
@click.pass_context
def cli(ctx, device):
    """PyTorch ViT 실험을 위한 CLI 도구. 모든 하위 명령어에 적용되는 옵션을 설정할 수 있습니다."""
    # 컨텍스트 객체를 생성하여 하위 명령어에 값을 전달합니다.
    ctx.ensure_object(dict)
    
    if device:
        ctx.obj['device'] = torch.device(device)
    else:
        # device 옵션이 없으면 자동으로 선택
        ctx.obj['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@cli.command(name='scenario-a', help="시나리오 A: 표준 사전학습 및 파인튜닝을 실행합니다.")
@click.option('--data-dir', default='./data', show_default=True, help="데이터셋을 저장할 디렉토리")
@click.option('--batch-size', default=512, show_default=True, help="학습 및 평가 시 배치 크기")
@click.option('--num-workers', default=4, show_default=True, help="데이터 로더 워커 수")
@click.option('--pretrain-epochs', default=10, show_default=True, help="사전학습 에포크 수")
@click.option('--finetune-epochs', default=20, show_default=True, help="파인튜닝 에포크 수")
@click.option('--pretrain-lr', default=1e-4, show_default=True, help="사전학습 학습률")
@click.option('--finetune-lr', default=5e-5, show_default=True, help="파인튜닝 학습률")
@click.option('--seed', default=42, show_default=True, help="재현성을 위한 랜덤 시드")
@click.option('--use-compile/--no-compile', default=True, show_default=True, help="torch.compile 사용 여부")
@click.pass_context
def run_scenario_a(ctx, data_dir, batch_size, num_workers, pretrain_epochs, finetune_epochs, pretrain_lr, finetune_lr, seed, use_compile):
    """시나리오 A 실행 함수"""
    setup_logging("log_scenario_a.txt")
    set_seed(seed)
    device = ctx.obj['device'] # 상위 cli 명령어에서 설정된 device 값을 가져옴
    logging.info(f"Using device: {device}")
    
    # locals()는 ctx를 포함하므로, 로깅 전에 제거하여 깔끔하게 만듭니다.
    params = locals().copy()
    del params['ctx']
    logging.info(f"Parameters: {params}")


    # 데이터 준비
    CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    base_train_dataset = CIFAR100(root=data_dir, train=True, download=True)
    base_test_dataset = CIFAR100(root=data_dir, train=False, download=True)
    train_indices = [i for i, label in enumerate(base_train_dataset.targets) if label < 80]
    unseen_train_indices = [i for i, label in enumerate(base_train_dataset.targets) if label >= 80]
    unseen_test_indices = [i for i, label in enumerate(base_test_dataset.targets) if label >= 80]

    # [A-1] 사전학습
    logging.info("\n" + "="*50 + "\n[A-1] Pre-training on 80 classes...\n" + "="*50)
    model = StandardViT(num_classes=80).to(device)
    if USE_TORCH_COMPILE and use_compile:
        logging.info("Using torch.compile() for model optimization.")
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=pretrain_lr)
    criterion = nn.CrossEntropyLoss()
    pretrain_full_dataset = Subset(CIFAR100(root=data_dir, train=True, transform=transform_train), train_indices)
    train_size = int(0.9 * len(pretrain_full_dataset))
    pretrain_train_split, pretrain_val_split = random_split(pretrain_full_dataset, [train_size, len(pretrain_full_dataset) - train_size])
    pretrain_loader = DataLoader(pretrain_train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    pretrain_val_loader = DataLoader(pretrain_val_split, batch_size=batch_size, num_workers=num_workers)
    
    pretrain_start_time = time.time()
    train_standard(model, pretrain_loader, pretrain_val_loader, criterion, optimizer, device, pretrain_epochs)
    pretrain_time = time.time() - pretrain_start_time

    # [A-2] 파인튜닝
    logging.info("\n" + "="*50 + "\n[A-2] Fine-tuning on 20 unseen classes...\n" + "="*50)
    # torch.compile 사용 시, 모델 구조 변경 후 다시 컴파일 필요
    if USE_TORCH_COMPILE and use_compile:
        model._orig_mod.classifier = nn.Linear(512, 20).to(device)
        model = torch.compile(model)
    else:
        model.classifier = nn.Linear(512, 20).to(device)
        
    finetune_full_dataset = RemappedDataset(base_train_dataset, unseen_train_indices, transform_train, label_offset=80)
    train_size_ft = int(0.9 * len(finetune_full_dataset))
    finetune_train_split, finetune_val_split = random_split(finetune_full_dataset, [train_size_ft, len(finetune_full_dataset) - train_size_ft])
    finetune_loader = DataLoader(finetune_train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    finetune_val_loader = DataLoader(finetune_val_split, batch_size=batch_size, num_workers=num_workers)
    optimizer_finetune = AdamW(model.parameters(), lr=finetune_lr)
    
    finetune_start_time = time.time()
    train_standard(model, finetune_loader, finetune_val_loader, criterion, optimizer_finetune, device, finetune_epochs)
    finetune_time = time.time() - finetune_start_time

    # [A-3] 최종 평가
    logging.info("\n" + "="*50 + "\n[A-3] Final Evaluation on Test Set...\n" + "="*50)
    eval_dataset = RemappedDataset(base_test_dataset, unseen_test_indices, transform_test, label_offset=80)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers)
    acc_a_final = evaluate_standard(model, eval_loader, device)

    # 최종 결과 요약
    logging.info("\n\n" + "="*50 + "\n📊 SCENARIO A: FINAL RESULTS SUMMARY\n" + "="*50)
    logging.info(f"  - Pre-training Time: {pretrain_time:.2f}s")
    logging.info(f"  - Fine-tuning Time:  {finetune_time:.2f}s")
    logging.info(f"  ▶ Final Test Accuracy: {acc_a_final:.2f}%")
    logging.info("="*50)


@cli.command(name='scenario-b', help="시나리오 B: PFN 기반 메타러닝을 실행합니다.")
@click.option('--data-dir', default='./data', show_default=True, help="데이터셋을 저장할 디렉토리")
@click.option('--batch-size', default=16, show_default=True, help="메타 학습 시 배치 크기 (에피소드 수)")
@click.option('--num-workers', default=4, show_default=True, help="데이터 로더 워커 수")
@click.option('--epochs', default=1000, show_default=True, help="메타 학습 에포크 수")
@click.option('--lr', default=1e-4, show_default=True, help="메타 학습 학습률")
@click.option('--n-way', default=5, show_default=True, help="N-way (클래스 수)")
@click.option('--k-shot', default=5, show_default=True, help="K-shot (서포트 샘플 수)")
@click.option('--q-queries', default=5, show_default=True, help="쿼리 샘플 수")
@click.option('--seed', default=42, show_default=True, help="재현성을 위한 랜덤 시드")
@click.option('--use-compile/--no-compile', default=True, show_default=True, help="torch.compile 사용 여부")
@click.pass_context
def run_scenario_b(ctx, data_dir, batch_size, num_workers, epochs, lr, n_way, k_shot, q_queries, seed, use_compile):
    """시나리오 B 실행 함수"""
    setup_logging("log_scenario_b.txt")
    set_seed(seed)
    device = ctx.obj['device'] # 상위 cli 명령어에서 설정된 device 값을 가져옴
    logging.info(f"Using device: {device}")
    
    params = locals().copy()
    del params['ctx']
    logging.info(f"Parameters: {params}")

    # 데이터 준비
    CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    transform_pfn_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandAugment(num_ops=2, magnitude=14), transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD), transforms.RandomErasing(p=0.5)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    base_train_dataset = CIFAR100(root=data_dir, train=True, download=True)
    base_test_dataset = CIFAR100(root=data_dir, train=False, download=True)
    train_indices = [i for i, label in enumerate(base_train_dataset.targets) if label < 80]
    unseen_test_indices = [i for i, label in enumerate(base_test_dataset.targets) if label >= 80]

    # [B-1] 메타학습
    logging.info("\n" + "="*50 + "\n[B-1] Meta-learning on 80 classes...\n" + "="*50)
    model = PosteriorInferenceViT(n_way=n_way).to(device)
    if USE_TORCH_COMPILE and use_compile:
        logging.info("Using torch.compile() for model optimization.")
        model = torch.compile(model)
        
    optimizer = AdamW(model.parameters(), lr=lr)
    meta_train_base = CIFAR100(root=data_dir, train=True, transform=transform_pfn_train)
    meta_train_dataset = Cifar100EpisodeDataset(Subset(meta_train_base, train_indices), n_way, k_shot, q_queries, num_episodes=500)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    meta_train_start_time = time.time()
    train_pfn(model, meta_train_loader, optimizer, device, epochs)
    meta_train_time = time.time() - meta_train_start_time

    # [B-2] 최종 평가
    logging.info("\n" + "="*50 + "\n[B-2] Final Evaluation on Test Set (Few-shot)...\n" + "="*50)
    meta_test_base = CIFAR100(root=data_dir, train=False, transform=transform_test)
    meta_test_dataset = Cifar100EpisodeDataset(Subset(meta_test_base, unseen_test_indices), n_way, k_shot, q_queries, num_episodes=500)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size=batch_size)
    acc_b_final = evaluate_pfn(model, meta_test_loader, device)

    # 최종 결과 요약
    logging.info("\n\n" + "="*50 + "\n📊 SCENARIO B: FINAL RESULTS SUMMARY\n" + "="*50)
    logging.info(f"  - Meta-learning Time: {meta_train_time:.2f}s")
    logging.info(f"  ▶ Final Few-shot Test Accuracy: {acc_b_final:.2f}% (No fine-tuning)")
    logging.info("="*50)

if __name__ == '__main__':
    cli()
