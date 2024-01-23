from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_, to_2tuple

class ConvEmbed(nn.Module):
    '''
    img/token map to Conv Embedding
    '''
    
    def __init__(self,
                 patch_size=7, # [7, 3, 3]
                 in_chans=3,   # [3, dim of stage1, dim of stage2]
                 embed_dim=64, # [64, 192, 384]
                 stride=4,     # [4, 2, 2]
                 padding=2,    # [2, 1, 1]
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
    
    
class AttentionConv(nn.Module):
    def __init__(self,
                 dim=64,        # [64,192,384]
                 num_heads=4,   # paper: [1,3,6], me: [4,8,16]
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 kernel_size=3,
                 padding_q=1,
                 padding_kv=1,
                 stride_q=1,
                 stride_kv=2,
                 **kwargs
                 ):
        super().__init__()
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        self.dim = dim
        self.num_heads = num_heads        
        self.scale = dim ** -0.5
        
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
        
        self.linear_proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        
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
            ('conv', nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=dim)),
            ('bn', nn.BatchNorm2d(dim)),
            ('rearrange', Rearrange('b c h w -> b (h w) c'))
        ]))
        
        return proj
    
    def forward(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        q = self.conv_proj_q(x)
        k = self.conv_proj_k(x)
        v = self.conv_proj_v(x)
        
        q = rearrange(self.linear_proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.linear_proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.linear_proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        
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
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class Block(nn.Module):
    
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=QuickGELU,
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
                 act_layer=QuickGELU,
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
                 act_layer=QuickGELU,
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
    
