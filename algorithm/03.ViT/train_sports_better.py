import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
import math
import time       
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


img_size = 224
patch_size = 16
num_classes = 100
dropout = 0.1

batch_size = 512

label_smoothing = 0.1
learning_rate = 1e-3
epochs = 100 

device = 'cuda:5'
model_path = 'sports.pth'  # 모델 저장 경로

# 데이터셋 경로 설정
data_dir = './data/sports'  # Tiny ImageNet 데이터셋이 저장된 경로

# Transforms 정의하기
train_transform = transforms.Compose([
    transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.9, scale=(0.02, 0.33)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset load
train_data = ImageFolder('./data/sports/train', transform=train_transform)
valid_data = ImageFolder('./data/sports/valid', transform=test_transform)
test_data = ImageFolder('./data/sports/test', transform=test_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class PatchEmbedding(nn.Module):
    """
    이미지를 패치로 분할하고, 각 패치를 임베딩하는 클래스입니다.

    Attributes:
    patch_size (int): 패치의 크기 (예: 16x16).
    n_patches (int): 이미지당 생성되는 패치의 총 수.
    projection (nn.Conv2d): 패치를 임베딩 벡터로 변환하는 컨볼루션 레이어.

    Args:
    img_size (int): 입력 이미지의 크기 (예: 32x32).
    patch_size (int): 패치의 크기 (예: 16x16).
    in_channels (int): 입력 이미지의 채널 수 (RGB의 경우 3).
    embed_dim (int): 임베딩 차원의 크기.
    """
    def __init__(self, img_size:int=32, patch_size:int=2, in_chans:int=3, embed_dim:int=768):
        super().__init__()
        
        # Patch 정보 인식
        self.patch_size = patch_size
        assert img_size % patch_size == 0, f'img size({img_size})는 patch size({patch_size})로 나뉘어야 합니다.'
        self.n_patches = (img_size // patch_size) ** 2

        # 컨볼루션을 사용하여 패치를 임베딩 벡터로 변환
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
         

    def forward(self, x):
        # x: [배치 크기, 채널 수, 높이, 너비]
        x = self.projection(x)  # 컨볼루션을 통한 임베딩: [배치 크기, 임베딩 차원, 패치 수, _]
        x = x.flatten(2)        # 평탄화: [배치 크기, 임베딩 차원, 패치 수]
        x = x.transpose(1, 2)   # 변환: [배치 크기, 패치 수, 임베딩 차원]
        
        return x
    
class PositionalEmbedding(nn.Module):
    """
    위치 임베딩을 추가하는 클래스입니다. 각 패치에 대한 위치 정보를 제공합니다.

    Attributes:
    scale (torch.Tensor): 스케일링 펙터.
    position_embedding (torch.nn.Parameter): 학습 가능한 위치 인코딩.

    Args:
    num_patches (int): 이미지당 생성되는 패치의 수.
    embed_dim (int): 임베딩 차원의 크기.
    """
    def __init__(self, num_patches:int, embed_dim:int):
        super().__init__()
        # self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim)) # [1, 패치 수+1, 임베딩 차원]

    def forward(self, x):
        # x = self.scale*x + self.position_embedding # scaled x에 위치 정보를 임베딩에 더함 
        x += self.position_embedding  
        return x # [배치 크기, 패치 수+1, 임베딩 차원]

# New Method 1 : transformer block에 작은 스케일 인자 곱하기
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return self.gamma * x

# New Method 2 : DropPath
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff input dims
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# New Method 3 : 개선된 Attention with fused attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# 
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., drop_path=0., qkv_bias=False, qk_norm=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ls1 = LayerScale(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ls2 = LayerScale(embed_dim)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., qk_norm=False):
        super().__init__()
        # 이미지 패치 임베딩
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer 블록
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Drop path rates
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i], qkv_bias=qkv_bias, qk_norm=qk_norm
            ) for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.init_weights()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# 모델 정의
vit = VisionTransformer(img_size=img_size, 
                        patch_size=patch_size, 
                        num_classes=num_classes, 
                        drop_rate=dropout,
                        attn_drop_rate=dropout,
                        drop_path_rate=dropout) 

# # 모델 초기화
# vit.apply(vit._init_weights)

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, t_max, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.t_max = t_max
        self.t_cur = 0
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:  # during warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:  # post warmup
            self.T_cur = (self.last_epoch - self.warmup_steps) % self.t_max
            cosine_decay = 0.5 * (1 + math.cos(math.pi * self.T_cur / self.t_max))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]
            
        
total_steps = len(train_loader) * epochs
warmup_steps = min(total_steps * 0.2, 10000)
print(f"\nWarmUp Step is {int(warmup_steps)}(epoch:{int(warmup_steps//len(train_loader)+1)}) of Total Step {total_steps}\n")

criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# # original Paper
optimizer = optim.Adam(vit.parameters(), lr=learning_rate, betas=[0.9,0.999], weight_decay=0.03)
# scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps, total_steps, t_max=total_steps//3)

# Method for Small Dataset
# optimizer = optim.Adam(vit.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)*5, gamma=0.9)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

training_time = 0
# early_stopping = EarlyStopping(patience=30)
losses = []
val_losses = []
lrs = []
best_loss = float('inf')

vit_save = False
vit.to(device)

# GradScaler 초기화
scaler = GradScaler()

for epoch in range(epochs):
    vit.train()
    start_time = time.time()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
    
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        # AutoCast 적용
        with autocast():
            outputs = vit(inputs)
            loss = criterion(outputs, labels)

        # Scaled Backward & Optimizer Step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        running_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    
    cur_step = len(train_loader) * epoch
    val_loss = -1
    if cur_step > warmup_steps:    
        vit.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = vit(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            vit_save = True
            torch.save(vit.state_dict(), model_path)

    epoch_duration = time.time() - start_time
    training_time += epoch_duration
    
    text = f'\tLoss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
    
    if vit_save:
        text += f' - model saved!'
        print(text)
        vit_save = False
    # elif epoch % 5 == 4 :
    #     print(text)
    else : 
        print(text)

    # # Early Stopping 체크
    # early_stopping(loss)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break
    
torch.save(vit.state_dict(), './last_sports.pth')

# 예측 수행 및 레이블 저장
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 혼동 행렬 생성
cm = confusion_matrix(all_labels, all_preds)

# 예측과 실제 레이블
y_true = all_labels  # 실제 레이블
y_pred = all_preds  # 모델에 의해 예측된 레이블

# 전체 데이터셋에 대한 정확도
accuracy = accuracy_score(y_true, y_pred)

# 평균 정밀도, 리콜, F1-Score ('weighted')
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# 판다스 데이터프레임으로 결과 정리
performance_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1_score]
})

# 데이터프레임 출력
print(performance_metrics)