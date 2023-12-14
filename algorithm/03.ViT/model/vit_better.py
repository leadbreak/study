import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.scale*x + self.position_embedding # scaled x에 위치 정보를 임베딩에 더함 
        # x += self.position_embedding  
        x = self.norm(x)
        return x # [배치 크기, 패치 수+1, 임베딩 차원]

# 파라미터 수 비교를 위해 가져온 이전 구현체
class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout, qkv_bias:bool=True, fused_attention:bool=True):
        super().__init__()
        self.fused_attention = fused_attention
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model ({d_model})은 n_heads ({n_heads})로 나누어 떨어져야 합니다.'

        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** 0.5

        # 쿼리, 키, 값에 대한 결합된 선형 변환
        self.fc_qkv = nn.Linear(d_model, d_model * 3, bias=qkv_bias)
        self.fc_o = nn.Linear(d_model, d_model)
        
        # normalize
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        # dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # batch_size = x.shape[0]
        B, N, C = x.shape

        # 결합된 선형 변환 수행
        qkv = self.fc_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 분리
        q, k = self.q_norm(q), self.k_norm(k) # Norm
        
        if self.fused_attention:
            attention = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p
            )
        else :            
            # 스케일드 닷-프로덕트 어텐션 계산
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attention = attn @ v

        # 어텐션 헤드 재조립 및 최종 선형 변환
        x = attention.transpose(1, 2).reshape(B, -1, self.d_model)
        x = self.fc_o(x)
        x = self.proj_drop(x)
        return x
    
# New Method 1 : transformer block에 작은 스케일 인자 곱하기
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
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
    def __init__(self, embed_dim:int, num_heads:int, mlp_ratio:float=4.0, dropout:float=0.1, estimate_params:bool=False, fused_attention:bool=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ls1 = LayerScale(embed_dim)
        self.ls2 = LayerScale(embed_dim)
        
        # self.drop_path = DropPath(dropout) if dropout > .0 else nn.Identity()
        self.drop_path = nn.Identity()
        self.estimate_params = estimate_params
        if estimate_params:
            self.attn = MHA(embed_dim, num_heads, dropout, fused_attention=fused_attention)
        else :
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) # [attention_output, attention weights]        

        mlp_hidden_dim = int(mlp_ratio * embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 멀티헤드 어텐션과 피드포워드 네트워크를 적용
        x = self.norm1(x)
        if self.estimate_params:
            x = x + self.drop_path(self.ls1(self.attn(x)))
        else :
            x = x + self.drop_path(self.ls1(self.attn(x, x, x)[0])) # attention output만 사용
        
        x2 = self.norm2(x)
        x = x + self.drop_path(self.ls2(self.mlp(x2)))
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
                 estimate_params:bool=False, fused_attention:bool=True):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout,estimate_params, fused_attention) 
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
        
    # 파라미터 초기화
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

    def forward(self, x):
        x = self.patch_embed(x)  # 이미지를 패치로 분할하고 임베딩
        x = self.pos_embed(x)    # 위치 인코딩 적용
        x = self.dropout(x)      # 임베딩 작업 후 dropout

        for layer in self.transformer_encoders:
            x = layer(x)  # 각 Transformer Encoder 레이어 적용

        x = self.norm(x)        # 정규화
        # x = self.dropout(x)     # dropout 적용
        
        # cls_token의 출력을 사용하여 분류
        cls_token_output = x[:, 0]  # 첫 번째 토큰 (cls_token) 추출
        x = self.head(cls_token_output)  # 최종 분류를 위한 선형 레이어
        return x