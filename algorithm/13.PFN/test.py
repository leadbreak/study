import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.transforms import functional as F_vision
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import random
import time
import sys

from einops.layers.torch import Rearrange

# --- 헬퍼 클래스: 로거 ---
class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding='utf-8')
    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
    def __del__(self):
        if not self.log_file.closed:
            self.log_file.close()

# --- [공통] ViT 모델 아키텍처 (이전과 동일) ---
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

# --- [시나리오 A] 헬퍼 함수 (수정됨) ---
class StandardViT(nn.Module):
    def __init__(self, num_classes, emb_size=512, num_layers=6):
        super().__init__()
        self.backbone = VisionTransformerEncoder(emb_size=emb_size, num_layers=num_layers)
        self.classifier = nn.Linear(emb_size, num_classes)
    def forward(self, x): return self.classifier(self.backbone(x))

def train_standard(model, train_loader, eval_loader, criterion, optimizer, device, epochs, logger):
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
        
        # [수정] 매 에포크 종료 후 성능 평가 및 로그 기록
        epoch_time = time.time() - epoch_start_time
        val_acc = evaluate_standard(model, eval_loader, device)
        logger.log(f"  Epoch {epoch+1}/{epochs} -> Val Accuracy: {val_acc:.2f}%, Time: {epoch_time:.2f}s")

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

# --- [시나리오 B] 헬퍼 함수 ---
class GaussianNoise(nn.Module):
    """
    훈련 시 입력 텐서에 가우시안 노이즈를 추가하는 모듈입니다.
    생성된 노이즈는 특정 범위로 클리핑하여 극단적인 값 발생을 방지합니다.
    """
    def __init__(self, stddev=1e-3):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        # 훈련 모드이고 stddev가 0이 아닐 때만 노이즈를 추가합니다.
        if self.training and self.stddev != 0:
            # 1. 표준 정규 분포에서 노이즈를 샘플링합니다.
            random_multiplier = 3 * (torch.rand(1, device=x.device) * 2 - 1)  # -4에서 4 사이의 값
            noise = torch.randn_like(x) * self.stddev * random_multiplier
            
            # 2. 노이즈를 입력 텐서에 더해줍니다.
            return x + noise
        
        # 훈련 모드가 아니거나 stddev가 0일 경우 원본 텐서를 그대로 반환합니다.
        return x


class PosteriorInferenceViT(nn.Module):
    def __init__(self, n_way, emb_size=512, num_heads=8, num_backbone_layers=6, num_inference_layers=4, noise_stddev=1e-3, add_noise=True):
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

class Cifar100EpisodeDataset(Dataset):
    def __init__(self, dataset, n_way, k_shot, q_queries):
        self.dataset, self.n_way, self.k_shot, self.q_queries = dataset, n_way, k_shot, q_queries
        self.targets_np = np.array(dataset.dataset.targets)[dataset.indices] if isinstance(dataset, Subset) else np.array(dataset.targets)
        self.label_to_indices = {label: np.where(self.targets_np == label)[0] for label in np.unique(self.targets_np)}
    def __len__(self): return 500
    def __getitem__(self, index):
        classes = random.sample(list(self.label_to_indices.keys()), self.n_way)
        s_imgs, s_lbls, q_imgs, q_lbls = [], [], [], []
        for i, cls in enumerate(classes):
            s_indices, q_indices = random.sample(list(self.label_to_indices[cls]), self.k_shot + self.q_queries)[:self.k_shot], random.sample(list(self.label_to_indices[cls]), self.k_shot + self.q_queries)[self.k_shot:]
            s_imgs.extend([self.dataset[j][0] for j in s_indices]); s_lbls.extend([i] * self.k_shot)
            q_imgs.extend([self.dataset[j][0] for j in q_indices]); q_lbls.extend([i] * self.q_queries)
        return (torch.stack(s_imgs), torch.LongTensor(s_lbls), torch.stack(q_imgs), torch.LongTensor(q_lbls))

def train_pfn(model, dataloader, optimizer, device, epochs, logger):
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
        
        # [수정] 매 에포크 종료 후 성능 및 시간 로그 기록
        epoch_time = time.time() - epoch_start_time
        avg_epoch_acc = 100 * total_acc_epoch / len(dataloader)
        logger.log(f"  Epoch {epoch+1}/{epochs} -> Meta-Train Accuracy: {avg_epoch_acc:.2f}%, Time: {epoch_time:.2f}s")

def evaluate_pfn(model, dataloader, device):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for s_img, s_lbl, q_img, q_lbl in tqdm(dataloader, desc="PFN Meta-evaluation"):
            s_img, s_lbl, q_img, q_lbl = map(lambda x: x.to(device), [s_img, s_lbl, q_img, q_lbl])
            total_acc += (model(s_img, s_lbl, q_img).argmax(dim=-1) == q_lbl).float().mean().item()
    return 100 * total_acc / len(dataloader) if len(dataloader) > 0 else 0

class RemappedDataset(Dataset):
    def __init__(self, original_dataset, indices, transform, label_offset):
        self.data, self.targets = [original_dataset.data[i] for i in indices], [original_dataset.targets[i] - label_offset for i in indices]
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.transform(F_vision.to_pil_image(self.data[idx])), self.targets[idx]

# --- 🚀 메인 실행 블록 ---
if __name__ == '__main__':
    logger = Logger("log.txt")
    
    # --- 1. 설정 ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE, NUM_WORKERS = 512, 64
    PRETRAIN_EPOCHS_A, FINETUNE_EPOCHS_A, META_EPOCHS_B = 1, 2, 1000
    N_WAY, K_SHOT, Q_QUERIES = 5, 5, 5
    
    logger.log(f"Using device: {DEVICE}\n" + "-"*55 + "\n" + f"BATCH_SIZE: {BATCH_SIZE}, NUM_WORKERS: {NUM_WORKERS}\n" +
               f"SCENARIO A: Pre-train={PRETRAIN_EPOCHS_A}, Fine-tune={FINETUNE_EPOCHS_A}\n" +
               f"SCENARIO B: Meta-train={META_EPOCHS_B}, N_WAY={N_WAY}, K_SHOT={K_SHOT}\n" + "-"*55)

    # --- 2. 데이터 준비 ---
    CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    transform_standard_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    transform_pfn_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandAugment(num_ops=2, magnitude=14), transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD), transforms.RandomErasing(p=0.5)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    base_train_dataset = CIFAR100(root='./data', train=True, download=True)
    base_test_dataset = CIFAR100(root='./data', train=False, download=True)
    train_indices = [i for i, label in enumerate(base_train_dataset.targets) if label < 80]
    unseen_train_indices = [i for i, label in enumerate(base_train_dataset.targets) if label >= 80]
    unseen_test_indices = [i for i, label in enumerate(base_test_dataset.targets) if label >= 80]
    
    # --- 3. ✅ 시나리오 A 실행 ---
    logger.log("\n" + "="*50 + "\n🚀 SCENARIO A: Standard Pre-training & Fine-tuning\n" + "="*50)
    
    # [A-1] 사전학습
    logger.log("\n[A-1] Pre-training on 80 classes...")
    model_a = StandardViT(num_classes=80).to(DEVICE)
    # model_a = torch.compile(model_a, mode='reduce-overhead')
    optimizer_a = AdamW(model_a.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    pretrain_full_dataset = Subset(CIFAR100(root='./data', train=True, transform=transform_standard_train), train_indices)
    # 검증 데이터셋 분리
    train_size = int(0.9 * len(pretrain_full_dataset))
    val_size = len(pretrain_full_dataset) - train_size
    pretrain_train_split, pretrain_val_split = random_split(pretrain_full_dataset, [train_size, val_size])
    pretrain_loader_a = DataLoader(pretrain_train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    pretrain_val_loader = DataLoader(pretrain_val_split, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    total_time_a1 = time.time()
    train_standard(model_a, pretrain_loader_a, pretrain_val_loader, criterion, optimizer_a, DEVICE, PRETRAIN_EPOCHS_A, logger)
    total_time_a1 = time.time() - total_time_a1
    
    # [A-2] 파인튜닝
    logger.log("\n[A-2] Fine-tuning on 20 unseen classes...")
    model_a.classifier = nn.Linear(512, 20).to(DEVICE)
    finetune_full_dataset = RemappedDataset(base_train_dataset, unseen_train_indices, transform_standard_train, label_offset=80)
    train_size_ft = int(0.9 * len(finetune_full_dataset))
    val_size_ft = len(finetune_full_dataset) - train_size_ft
    finetune_train_split, finetune_val_split = random_split(finetune_full_dataset, [train_size_ft, val_size_ft])
    finetune_loader = DataLoader(finetune_train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    finetune_val_loader = DataLoader(finetune_val_split, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    optimizer_finetune = AdamW(model_a.parameters(), lr=5e-5)
    
    total_time_a2 = time.time()
    train_standard(model_a, finetune_loader, finetune_val_loader, criterion, optimizer_finetune, DEVICE, FINETUNE_EPOCHS_A, logger)
    total_time_a2 = time.time() - total_time_a2
    
    # [A-3] 최종 평가
    logger.log("\n[A-3] Final Evaluation on Test Set...")
    eval_dataset = RemappedDataset(base_test_dataset, unseen_test_indices, transform_test, label_offset=80)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    acc_a_final = evaluate_standard(model_a, eval_loader, DEVICE)
    
    # --- 4. ✅ 시나리오 B 실행 ---
    logger.log("\n" + "="*50 + "\n🧠 SCENARIO B: PFN-based Meta-Learning\n" + "="*50)
    
    # [B-1] 메타학습
    logger.log("\n[B-1] Meta-learning on 80 classes...")
    model_b = PosteriorInferenceViT(n_way=N_WAY).to(DEVICE)
    # model_b = torch.compile(model_b, mode='reduce-overhead')
    optimizer_b = AdamW(model_b.parameters(), lr=1e-4)
    meta_train_base = CIFAR100(root='./data', train=True, transform=transform_pfn_train)
    meta_train_dataset = Cifar100EpisodeDataset(Subset(meta_train_base, train_indices), N_WAY, K_SHOT, Q_QUERIES)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    
    total_time_b1 = time.time()
    train_pfn(model_b, meta_train_loader, optimizer_b, DEVICE, META_EPOCHS_B, logger)
    total_time_b1 = time.time() - total_time_b1

    # [B-2] 최종 평가
    logger.log("\n[B-2] Final Evaluation on Test Set (Few-shot)...")
    meta_test_base = CIFAR100(root='./data', train=False, transform=transform_test)
    meta_test_dataset = Cifar100EpisodeDataset(Subset(meta_test_base, unseen_test_indices), N_WAY, K_SHOT, Q_QUERIES)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size=16)
    acc_b_final = evaluate_pfn(model_b, meta_test_loader, DEVICE)
    
    # --- 5. 📊 최종 결과 요약 ---
    logger.log("\n\n" + "="*50 + "\n📊 FINAL RESULTS SUMMARY\n" + "="*50)
    logger.log("[Scenario A: Standard Method]")
    logger.log(f"  - Pre-training Time: {total_time_a1:.2f}s")
    logger.log(f"  - Fine-tuning Time:  {total_time_a2:.2f}s")
    logger.log(f"  ▶ Final Test Accuracy: {acc_a_final:.2f}%")
    logger.log("-" * 50)
    logger.log("[Scenario B: PFN Method]")
    logger.log(f"  - Meta-learning Time: {total_time_b1:.2f}s")
    logger.log(f"  ▶ Final Few-shot Test Accuracy: {acc_b_final:.2f}% (No fine-tuning)")
    logger.log("="*50)