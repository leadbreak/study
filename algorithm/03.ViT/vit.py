import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

import math
import matplotlib.pyplot as plt
import time

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
    def __init__(self, img_size:int=32, patch_size:int=2, in_channels:int=3, embed_dim:int=768):
        super().__init__()
        
        # Patch 정보 인식
        self.patch_size = patch_size
        assert img_size % patch_size == 0, f'img size({img_size})는 patch size({patch_size})로 나뉘어야 합니다.'
        self.n_patches = (img_size // patch_size) ** 2

        # 컨볼루션을 사용하여 패치를 임베딩 벡터로 변환
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # class token 추가
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))        

    def forward(self, x):
        # x: [배치 크기, 채널 수, 높이, 너비]
        x = self.projection(x)  # 컨볼루션을 통한 임베딩: [배치 크기, 임베딩 차원, 패치 수, _]
        x = x.flatten(2)        # 평탄화: [배치 크기, 임베딩 차원, 패치 수]
        x = x.transpose(1, 2)   # 변환: [배치 크기, 패치 수, 임베딩 차원]
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # [배치크기, 1, 임베딩 차원]
        x = torch.cat((cls_tokens, x), dim=1) # cls_token 추가: [배치크기, 패치 수+1, 임베딩 차원]
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
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim)) # [1, 패치 수+1, 임베딩 차원]

    def forward(self, x):
        x = self.scale*x + self.position_embedding # scaled x에 위치 정보를 임베딩에 더함 
        # x += self.position_embedding  
        return x # [배치 크기, 패치 수+1, 임베딩 차원]
    
# 파라미터 수 비교를 위해 가져온 이전 구현체
class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model ({d_model})은 n_heads ({n_heads})로 나누어 떨어져야 합니다.'

        self.head_dim = d_model // n_heads  # int 형변환 제거

        # 쿼리, 키, 값에 대한 선형 변환
        self.fc_q = nn.Linear(d_model, d_model) 
        self.fc_k = nn.Linear(d_model, d_model) 
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        # 어텐션 점수를 위한 스케일 요소
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        

    def forward(self, Q, K, V):
        batch_size = Q.shape[0]

        # 쿼리, 키, 값에 대한 선형 변환 수행
        Q = self.fc_q(Q) 
        K = self.fc_k(K)
        V = self.fc_v(V)

        # 멀티 헤드 어텐션을 위해 텐서 재구성 및 순서 변경
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 스케일드 닷-프로덕트 어텐션 계산
        attention_score = Q @ K.permute(0, 1, 3, 2) / self.scale

        # 소프트맥스를 사용하여 어텐션 확률 계산
        attention_dist = torch.softmax(attention_score, dim=-1)

        # 어텐션 결과
        attention = attention_dist @ V

        # 어텐션 헤드 재조립
        x = attention.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # 최종 선형 변환
        x = self.fc_o(x)

        return x
    
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder 레이어를 정의하는 클래스입니다.

    Attributes:
    norm1, norm2 (nn.LayerNorm): 정규화 레이어.
    attn (nn.MultiheadAttention): 멀티헤드 어텐션 레이어.
    mlp (nn.Sequential): 피드포워드 네트워크.

    Args:
    embed_dim (int): 임베딩 차원의 크기.
    num_heads (int): 멀티헤드 어텐션에서의 헤드 수.
    mlp_ratio (float): 첫 번째 선형 레이어의 출력 차원을 결정하는 비율.
    dropout (float): 드롭아웃 비율.
    """
    def __init__(self, embed_dim:int, num_heads:int, mlp_ratio:float=4.0, dropout:float=0.1, estimate_params:bool=False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.estimate_params = estimate_params
        if estimate_params:
            self.attn = MHA(embed_dim, num_heads)
        else :
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) # [attention_output, attention weights]        

        mlp_hidden_dim = int(mlp_ratio * embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 멀티헤드 어텐션과 피드포워드 네트워크를 적용
        if self.estimate_params:
            x = self.norm(x)
            x = x + self.attn(x, x, x)
        else :
            x = x + self.attn(x, x, x)[0] # attention output만 사용
        
        x2 = self.norm(x)
        x = x + self.mlp(x2)
        return x
    
class VisionTransformer(nn.Module):
    """
    전체 Vision Transformer 모델을 정의하는 클래스입니다.

    Attributes:
    patch_embed (PatchEmbedding): 이미지를 패치로 분할하고 임베딩하는 레이어.
    pos_embed (PositionalEncoding): 위치 인코딩 레이어.
    transformer_encoders (nn.ModuleList): Transformer Encoder 레이어들의 리스트.
    norm (nn.LayerNorm): 정규화 레이어.
    head (nn.Linear): 최종 분류를 위한 선형 레이어.

    Args:
    img_size (int): 입력 이미지의 크기 (예: 32x32).
    patch_size (int): 패치의 크기 (예: 16x16).
    in_channels (int): 입력 이미지의 채널 수 (RGB의 경우 3).
    num_classes (int): 분류할 클래스의 수 (CIFAR-10의 경우 10).
    embed_dim (int): 임베딩 차원의 크기.
    num_heads (int): 멀티헤드 어텐션에서의 헤드 수.
    num_layers (int): Transformer Encoder 레이어의 수.
    mlp_ratio (float): 피드포워드 네트워크의 차원 확장 비율.
    dropout (float): 드롭아웃 비율.
    """
    def __init__(self, img_size:int=32, patch_size:int=4, in_channels:int=3, 
                 num_classes:int=100, embed_dim:int=768, num_heads:int=12, 
                 num_layers:int=12, mlp_ratio:float=4., dropout:float=0.1,
                 estimate_params:bool=False):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout,estimate_params) 
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)  # 이미지를 패치로 분할하고 임베딩
        x = self.pos_embed(x)    # 위치 인코딩 적용
        x = self.dropout(x)      # 임베딩 작업 후 dropout

        for layer in self.transformer_encoders:
            x = layer(x)  # 각 Transformer Encoder 레이어 적용

        x = self.norm(x)        # 정규화
        x = self.dropout(x)     # dropout 적용
        
        # cls_token의 출력을 사용하여 분류
        cls_token_output = x[:, 0]  # 첫 번째 토큰 (cls_token) 추출
        x = self.head(cls_token_output)  # 최종 분류를 위한 선형 레이어
        return x
    
# 모델 테스트 - 임의의 이미지 데이터 생성 (B, C, W, H)
test_img = torch.randn(2, 3, 224, 224) 

# 모델 초기화
vit = VisionTransformer(img_size=224, 
                        patch_size=16, 
                        num_classes=100, 
                        dropout=0.0,
                        estimate_params=True) # 파라미터 수를 측정하고 싶으면 True
output = vit(test_img)     # 테스트 이미지를 모델에 통과


batch_size = 512
num_workers = 8
num_classes = 100

# label_smoothing = 0.1
learning_rate = 0.001
epochs = 300

device = 'cuda:3'
model_path = 'vit_cifar100.pth'  # 모델 저장 경로

# 데이터 증강을 위한 전처리
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # 50% 확률로 수평 뒤집기
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변경
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]
        

total_steps = len(trainloader) * epochs
warmup_steps = total_steps * 0.1

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters(), lr=learning_rate, betas=[0.9,0.999], weight_decay=0.03)
scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps, total_steps)

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
            
from tqdm import tqdm

training_time = 0
early_stopping = EarlyStopping(patience=int(epochs*0.1))
losses = []
val_losses = []
lrs = []
best_val_loss = float('inf')

vit_save = False
vit.to(device)

for epoch in range(epochs):
    vit.train()
    start_time = time.time()
    running_loss = 0.0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}")
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = vit(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
    epoch_loss = running_loss / len(trainloader)
    losses.append(epoch_loss)

    # 검증 손실 계산
    vit.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = vit(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(testloader)
    val_losses.append(val_loss)

    # 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        vit_save = True
        torch.save(vit.state_dict(), model_path)

    epoch_duration = time.time() - start_time
    training_time += epoch_duration
    if vit_save:
        print(f'\tLoss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec - model saved!')
        vit_save = False
    else :
        print(f'\tLoss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec')

    # Early Stopping 체크
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# # 학습 및 검증 손실 시각화
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Training & Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(lrs, label='Learning Rate')
# plt.title('Learning Rate')
# plt.xlabel('Batch number')
# plt.ylabel('Learning Rate')
# plt.legend()
# plt.show()

print("="*120)
print("\nFinished!!\n")
print("="*120)