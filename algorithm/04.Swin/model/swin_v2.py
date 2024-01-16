import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

class embeddings(nn.Module):
    def __init__(self, 
                 img_size: int | tuple = 224, 
                 patch_size: int = 4, 
                 in_chans: int = 3, 
                 embed_dim: int = 96, 
                 norm_layer=None, 
                 drop: float = 0.0):
        super().__init__()
        # 이미지 크기 설정. 튜플이 아닐 경우 to_2tuple 함수를 통해 튜플로 변환
        self.img_size = img_size if isinstance(img_size, tuple) else to_2tuple(img_size)
        # 패치 크기 설정
        self.patch_size = patch_size
        # 이미지를 패치로 나눈 해상도 계산 (이미지 크기를 패치 크기로 나눔)
        self.patches_resolution = [self.img_size[0] // patch_size, self.img_size[1] // patch_size]
        # 전체 패치의 수 계산
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        # 입력 채널 수와 임베딩 차원 설정
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 2D 합성곱 레이어를 사용하여 패치 임베딩 수행
        self.patch_embeddings = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)        
        # 정규화 레이어 설정. 주어진 norm_layer가 없을 경우 Identity 레이어 사용
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()            
        # 드롭아웃 레이어 설정
        self.dropout = nn.Dropout(drop)

    def forward(self, x):        
        x = self.patch_embeddings(x).flatten(2).transpose(1, 2)
        # 정규화 레이어 적용 (norm_layer가 Identity일 경우 아무 변화 없음)
        if self.norm is not None:
            x = self.norm(x)
        return x

# Cyclic shift 함수 정의
def cyclic_shift(img, shift_size):
    """ 이미지에 대해 cyclic shift를 수행하는 함수 """
    shifted_img = torch.roll(img, shifts=(-shift_size, -shift_size), dims=(1, 2))
    return shifted_img

def window_partition(x, window_size):
    """ 입력텐서(x)를 window_size 크기의 윈도우로 분할    
    
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows # (num_windows, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    """ window_partition을 통해 분할된 윈도우들을 원래 이미지로 복원
    
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x # (B, H, W, C)


class WindowAttention(nn.Module):
    """ 윈도우 기반 멀티헤드 셀프 어텐션 (W-MSA) 모듈. 시프트된 윈도우 지원.
    
    Args:
        dim (int): 입력 차원 수.
        window_size (tuple[int]): 윈도우의 높이와 너비.
        num_heads (int): 어텐션 헤드의 수.
        qkv_bias (bool, optional): QKV 선형 변환에 바이어스 추가 여부. 기본값: True.
        attn_drop (float, optional): 어텐션 가중치의 드롭아웃 비율. 기본값: 0.0.
        proj_drop (float, optional): 출력의 드롭아웃 비율. 기본값: 0.0.
        pretrained_window_size (tuple[int]): stage2 pre-training 단계에서 학습된 사전 윈도우 사이즈
        rpb_scale (bool, optional): Relative Positional Bias Table에 대한 스케일 여부
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=0, rpb_scale=True):
        super().__init__()
        self.dim = dim  # 입력 차원
        self.window_size = window_size  # 윈도우 크기
        self.pretrained_window_size = pretrained_window_size # 사전학습된 윈도우 크기
        self.rpb_scale = rpb_scale  # rpb table에 대한 스케일 여부
        self.num_heads = num_heads  # 어텐션 헤드 수
        
        # V2: t to scale cosine score
        self.t_scale = nn.Parameter(torch.log(10*torch.ones((num_heads, 1, 1))), requires_grad=True)

        # V2: small meta network to generate continuous relative position bias
        self.crpb_mlp = nn.Sequential(nn.Linear(2, 384, bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(p=.125),
                                      nn.Linear(384, num_heads, bias=False))

        # V2: generate and register rpb
        self._create_relative_position_table()        
        
        # QKV 선형 변환과 드롭아웃 레이어 초기화
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # QKV 행렬 생성
        self.attn_drop = nn.Dropout(attn_drop)  # 어텐션 드롭아웃
        self.proj = nn.Linear(dim, dim)  # 최종 출력을 위한 선형 변환
        self.proj_drop = nn.Dropout(proj_drop)  # 출력 드롭아웃

        self.softmax = nn.Softmax(dim=-1)  # 소프트맥스 함수

    # V2
    def _create_relative_position_table(self):
        '''
        [v1] 
        - 상대적 위치 바이어스 테이블이 학습가능한 파라미터로 취급되며, 특정분포로 초기화되어야 함
        - 상대적 위치 정보가 고정된 값으로, 학습 과정에서 변경되지 않고 별도로 저장할 필요 없음
        [v2] 
        - 상대적 위치 좌표를 로그 스케일로 변환하는 방식을 사용해, 윈도우 크기에 비례해 동적으로 계산하며, 초기화 불필요
        - 상대적 위치 정보가 학습가능하지 않지만 최종 저장되는 모델에 포함되어야 효율적이기 때문에 buffer로 등록
        '''
        # 상대적 위치 좌표 테이블 생성
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)

        # 윈도우 크기로 정규화 -> 형태 조정 방법에 따라 필요없을수도?
        if self.pretrained_window_size > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size - 1)
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        # 로그 스케일로 변환
        if self.rpb_scale: 
            relative_coords_table *= 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8))
        else:
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # 윈도우 내 각 토큰 쌍에 대한 상대적 위치 인덱스 계산
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)


    def forward(self, x, mask=None):
        # 순방향 전파 함수
        B_, N, C = x.shape  # 입력 텐서의 형태
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Q, K, V 분리
        
        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2,-1))
        t_scale = torch.clamp(self.t_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.t_scale.device)).exp()
        attn = attn * t_scale

        # v2: 상대적 위치 편향 적용
        relative_position_bias_table = self.crpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.rpb_scale:
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        # 마스크 적용 (필요한 경우)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # 어텐션 드롭아웃 적용

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 어텐션 적용 및 재구성
        x = self.proj(x)  # 최종 선형 변환
        x = self.proj_drop(x)  # 출력 드롭아웃 적용
        return x
    
class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# transformer block에 작은 스케일 인자 곱하기
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return self.gamma * x
    
class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block 클래스

    Args:
        dim (int): 입력 채널의 수.
        input_resolution (tuple[int]): 입력 해상도 (높이, 너비).
        num_heads (int): 어텐션 헤드의 수.
        window_size (int): 윈도우의 크기.
        shift_size (int): 윈도우 시프트 크기.
        mlp_ratio (float): MLP의 hidden dimension과 embedding dimension의 비율.
        qkv_bias (bool, optional): QKV에 바이어스 추가 여부.
        drop (float, optional): 드롭아웃 확률.
        attn_drop (float, optional): 어텐션 드롭아웃 확률.
        drop_path (float, optional): 드롭 패스 확률.
        act_layer (nn.Module, optional): 활성화 함수 레이어.
        norm_layer (nn.Module, optional): 정규화 레이어.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 윈도우 크기가 입력 해상도보다 크거나 같으면 시프트 크기를 0으로 조정
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size는 0과 window_size 사이의 값이어야 함"       
        
        # 윈도우 기반 멀티헤드 셀프 어텐션 레이어
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size)

        # stochastic depth + res-post-norm 1
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)        

        # Layer 5: MLP 레이어
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # stochastic depth + res-post-norm 2
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # 어텐션 마스크 계산
        self._calculate_attn_mask()

    def _calculate_attn_mask(self):
        # 시프트된 윈도우 기반의 어텐션을 위한 마스크 계산
        if (self.shift_size > 0) or (self.shift_size != self.window_size):
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 이미지 마스크 초기화
            h_slices = [slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)]
            w_slices = [slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)]
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # 순방향 전파
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"L({L}) != H({H}) x W({W})"

        shortcut = x
        x = x.view(B, H, W, C)

        # 순환 이동 및 윈도우 어텐션
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 윈도우 병합 및 순환 이동 복구
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)
        x = x.view(B, H * W, C)

        # res-post-norm 적용 및 레이어 통과 후 잔차 연결
        x = shortcut + self.norm1(self.drop_path1(x))
        x = x + self.norm2(self.drop_path2(self.mlp(x)))

        return x

class PatchMerging(nn.Module):
    """ 패치 병합 레이어: 인접한 4개의 패치를 하나로 병합

    Args:
        input_resolution (tuple[int]): 입력 해상도 (높이, 너비)
        dim (int): 입력 채널 수
        norm_layer (nn.Module, optional): 정규화 레이어
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 차원 감소를 위한 선형 레이어
        self.norm = norm_layer(4 * dim)  # 정규화 레이어

    def forward(self, x):
        """
        x: (B, H*W, C) 형태의 텐서
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "입력 특징의 크기가 올바르지 않음"
        assert H % 2 == 0 and W % 2 == 0, "H와 W는 2의 배수여야 함"

        # 패치를 재배열하여 4개의 그룹으로 나눔
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # 첫 번째 그룹
        x1 = x[:, 1::2, 0::2, :]  # 두 번째 그룹
        x2 = x[:, 0::2, 1::2, :]  # 세 번째 그룹
        x3 = x[:, 1::2, 1::2, :]  # 네 번째 그룹
        x = torch.cat([x0, x1, x2, x3], -1)  # 4개 그룹 연결
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C

        x = self.norm(x)  # 정규화
        x = self.reduction(x)  # 차원 감소

        return x

class StageLayer(nn.Module):
    """ Swin Transformer의 기본 레이어: 여러 Swin Transformer 블록과 패치 병합 레이어 포함

    Args:
        dim (int): 입력 채널 수
        input_resolution (tuple[int]): 입력 해상도 (높이, 너비)
        depth (int): 블록의 수
        num_heads (int): 어텐션 헤드의 수
        window_size (int): 윈도우의 크기
        mlp_ratio (float): MLP의 hidden dimension과 embedding dimension의 비율
        qkv_bias (bool, optional): QKV에 바이어스 추가 여부
        drop (float, optional): 드롭아웃 확률
        attn_drop (float, optional): 어텐션 드롭아웃 확률
        drop_path (float | tuple[float], optional): 드롭 패스 확률
        norm_layer (nn.Module, optional): 정규화 레이어
        downsample (nn.Module | None, optional): 다운샘플링 레이어
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Swin Transformer 블록들을 순차적으로 구성
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, 
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # 다운샘플링 레이어 (있을 경우)
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.downsample(x)
        return x
    
    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

class SwinTransformerV2(nn.Module):
    """ Swin Transformer V2 전체 모델
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True, pretrained_window_sizes=[0,0,0,0],
                 ape=True,**kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ape = ape

        # 패치 임베딩 레이어
        self.embeddings = embeddings(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        # absolute positional embedding
        if self.ape:
            num_patches = self.embeddings.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
            
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 스테이지별 레이어 생성
        self.stages = nn.ModuleList()
        for i_stage in range(len(depths)):
            layer = StageLayer(
                dim=int(embed_dim * 2 ** i_stage),
                input_resolution=(self.embeddings.patches_resolution[0] // (2 ** i_stage),
                                  self.embeddings.patches_resolution[1] // (2 ** i_stage)),
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_stage < len(depths) - 1 else None,
                pretrained_window_size=pretrained_window_sizes[i_stage])
            self.stages.append(layer)
            
        self.final_dim = embed_dim * 2 ** (len(depths) - 1)  # 최종 차원 계산
        self.layernorm = norm_layer(self.final_dim)  # 최종 출력 정규화 레이어
        self.pooler = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        self.classifier = nn.Linear(self.final_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 초기화
        self.apply(self._init_weights)
        for bly in self.stages:
            bly._init_respostnorm()
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                    
    def forward_features(self, x):
        x = self.embeddings(x)     
        if self.ape:
            x  = x + self.absolute_pos_embed
        x = self.pos_drop(x)
           
        for stage in self.stages:
            x = stage(x)            
        x = self.layernorm(x)             # (B, L, C)
        x = self.pooler(x.transpose(1,2)) # (B, C, 1)
        x = torch.flatten(x, 1)           # (B, C)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)   
        x = self.classifier(x)
        return x
    