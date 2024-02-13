'''
1,000epoch 中 900 : 0.9329
'''
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time

from timm.data import Mixup
import transformers

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_, to_2tuple

from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ConvEmbed(nn.Module):
    '''
    img/token map to Conv Embedding
    '''
    
    def __init__(self,
                 patch_size=11, # [11, 7, 7, 3]
                 in_chans=3,   # [3, dim of stage1, dim of stage2]
                 embed_dim=64, # [32, 64, 192, 384]
                 stride=4,     # [6, 4, 4, 2]
                 padding=2,    # [3, 2, 2, 1]
                 norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        x = self.proj(x)
        
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class AttentionConv(nn.Module):
    def __init__(self,
                 dim=64,        # [32,64,192,384]
                 num_heads=4,   # [1,3,6,9]
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 kernel_size=3,
                 padding_q=1,
                 padding_kv=1,
                 stride_q=1,
                 stride_kv=2,
                 act_layer=nn.GELU,
                 **kwargs
                 ):
        super().__init__()
        self.qkv_bias = qkv_bias
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        self.dim = dim
        self.num_heads = num_heads        
        self.scale = dim ** -0.5
        self.act_layer = act_layer()
        
        self.conv_proj_q = self._build_projection(dim,
                                                  kernel_size,
                                                  padding_q,
                                                  stride_q,
                                                  )
        self.conv_proj_k = self._build_projection(dim,
                                                  kernel_size,
                                                  padding_kv,
                                                  stride_kv,
                                                  )
        
        self.conv_proj_v = self._build_projection(dim,
                                                  kernel_size,
                                                  padding_kv,
                                                  stride_kv,
                                                  )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.linear_proj_last = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)        
        
    def _build_projection(self,
                          dim,
                          kernel_size,
                          padding,
                          stride,
                          ):
        
        proj = nn.Sequential(OrderedDict([
            ('depthwise', nn.Conv2d(
                dim,
                dim*4,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=self.qkv_bias,
                groups=dim)),
            ('rearrange1', Rearrange('b c h w -> b h w c')),
            ('ln', nn.LayerNorm(dim*4)),
            ('rearrange2', Rearrange('b h w c -> b c h w')),
            ('pointwise', nn.Conv2d(
                dim*4,
                dim,
                kernel_size=1,
                bias=self.qkv_bias)),
            ('activation', self.act_layer),
            ('rearrange3', Rearrange('b c h w -> b (h w) c')),

        ]))
        
        return proj
    
    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        q = F.normalize(self.conv_proj_q(x), dim=-1)
        k = F.normalize(self.conv_proj_k(x), dim=-1)
        v = self.conv_proj_v(x)
        
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = self.attn_drop(F.softmax(attn_score, dim=-1))
        
        x = torch.matmul(attn, v)
        batch_size, num_heads, seq_length, depth = x.size()
        x = x.view(batch_size, seq_length, num_heads * depth)
        
        x = self.proj_drop(self.linear_proj_last(x))
        
        return x

# transformer block에 작은 스케일 인자 곱하기
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return self.gamma * x
    
class Block(nn.Module):
    
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs
                ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim)
        self.attn = AttentionConv(dim=dim,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=drop,
                                  act_layer=act_layer,
                                  **kwargs)        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        
    def forward(self, x, h, w):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(self.ls1(attn))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs
                 ):
        
        super().__init__()

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            ) for _ in range(depth)
        ])

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        _, _, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.pos_drop(x)

        for _, blk in enumerate(self.blocks):
            x = blk(x, H, W)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
    
class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=100,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
                'stride_kv': spec['STRIDE_KV'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            
            self.stages.append(stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.pooler = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def forward_features(self, x):
        for stage in self.stages:
            x = stage(x)

        x = rearrange(x, 'b c h w -> b (h w) c') # (B, L, C)
        x = self.norm(x)                         # (B, L, C)
        x = self.pooler(x.transpose(1,2))        # (B, C, 1)
        x = torch.flatten(x, 1)                  # (B, C)
        # x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
spec = {
    'NUM_STAGES': 4,
    'PATCH_SIZE': [7,7,7,7],
    'PATCH_STRIDE': [4,2,2,2],
    'PATCH_PADDING': [2,3,3,3],
    'DIM_EMBED': [64,128,192,256],
    'DEPTH': [2,2,6,2],
    'NUM_HEADS': [4,8,12,16],   # original : [1,3,6]
    'MLP_RATIO': [4.,4.,4.,4.],
    'QKV_BIAS': [True, True, True, True],
    'DROP_RATE': [0.,0.,0.,0.],
    'ATTN_DROP_RATE': [0.,0.,0.,0.],
    'DROP_PATH_RATE': [0.1,0.1,0.1,0.1],
    'KERNEL_QKV': [7,7,7,7],
    'PADDING_Q': [3,3,3,3],
    'PADDING_KV': [3,3,3,3],
    'STRIDE_Q': [1,1,1,1],
    'STRIDE_KV': [2,2,2,2],
}

model = ConvolutionalVisionTransformer(act_layer=QuickGELU, spec=spec)

from torchsummary import summary

model_summary = summary(model.cuda(), (3, 224, 224))

print(model_summary)

# Transforms 정의하기
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=1., scale=(0.02, 0.33)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = '../data/sports'
batch_size = 256

train_path = data_dir+'/train'
valid_path = data_dir+'/valid'
test_path = data_dir+'/test'

# dataset load
train_data = ImageFolder(train_path, transform=train_transform)
valid_data = ImageFolder(valid_path, transform=test_transform)
test_data = ImageFolder(test_path, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = 'cuda:2'
max_norm = 3.0 

model.to(device)
model_path = '../models/cvt/model_revision.pth'

mixup_fn = Mixup(mixup_alpha=.8, 
                cutmix_alpha=1., 
                prob=1., 
                switch_prob=0.5, 
                mode='batch',
                label_smoothing=.1,
                num_classes=100)

epochs = 1000

criterion = nn.CrossEntropyLoss(label_smoothing=0.)
optimizer = optim.AdamW(model.parameters())
warmup_steps = int(len(train_loader)*(epochs)*0.1)
train_steps = len(train_loader)*(epochs)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=warmup_steps, 
                                                        num_training_steps=train_steps,
                                                        num_cycles=0.5)

training_time = 0
losses = []
val_losses = []
lrs = []
best_loss = float('inf')

# GradScaler 초기화
scaler = GradScaler()

for i in range(epochs // 100):
    for epoch in range(100):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1 + i*100}")
        
        for _, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = mixup_fn(inputs, labels)
            optimizer.zero_grad()

            # AutoCast 적용
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # 스케일링된 그라디언트 계산
            scaler.scale(loss).backward()

            # 그라디언트 클리핑 전에 스케일링 제거
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=max_norm)

            # 옵티마이저 스텝 및 스케일러 업데이트
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            lr = optimizer.param_groups[0]["lr"]
            lrs.append(lr)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)        

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(valid_loader)
        val_losses.append(val_loss)
        
        # 모델 저장
        total_loss = val_loss + epoch_loss
        if total_loss < best_loss:
            best_loss = total_loss
            model_save = True
            if model_save:
                torch.save(model.state_dict(), model_path)

        epoch_duration = time.time() - start_time
        training_time += epoch_duration
        
        text = f'\tLoss: {epoch_loss:.4f}, Val_Loss: {val_loss:.4f}, Total Mean Loss: {total_loss/2:.4f}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
        
        if model_save:
            text += f' - model saved!'
            model_save = False

        print(text)

    # 예측 수행 및 레이블 저장
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
    print(f"\n[{i*100+100} epoch result]\n", performance_metrics)

