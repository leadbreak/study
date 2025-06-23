import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")


class FeatureWiseAttention(nn.Module):
    """피처별 어텐션 메커니즘"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch_size, hidden_dim]
        attention_weights = self.attention(x)  # [seq_len, batch_size, 1]
        attention_weights = F.softmax(attention_weights, dim=0)
        return torch.sum(x * attention_weights, dim=0)  # [batch_size, hidden_dim]

class CrossAttention(nn.Module):
    """컨텍스트-쿼리 크로스 어텐션"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = key.size(1)
        
        # Multi-head projection
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.out_proj(attended)

# ----------------------------
# TabPFN V2 Model Architecture
# ----------------------------

class TabPFNV2(nn.Module):
    """TabPFN V2 메인 아키텍처"""
    
    def __init__(self, max_input_dim: int = 32, hidden_dim: int = 256, 
                 num_layers: int = 8, num_heads: int = 8, max_len: int = 512,
                 max_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        
        self.max_input_dim = max_input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.max_classes = max_classes
        
        # 입력 임베딩
        self.feature_embed = nn.Linear(max_input_dim, hidden_dim)
        self.label_embed = nn.Embedding(max_classes, hidden_dim)
        
        # 위치 임베딩
        self.position_embed = nn.Parameter(torch.randn(max_len, hidden_dim))
        
        # 타입 임베딩 (컨텍스트 vs 쿼리)
        self.type_embed = nn.Parameter(torch.randn(2, hidden_dim))
        
        # 피처 정규화
        self.feature_norm = nn.LayerNorm(max_input_dim)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 특수 어텐션 메커니즘
        self.feature_attention = FeatureWiseAttention(hidden_dim)
        self.cross_attention = CrossAttention(hidden_dim, num_heads)
        
        # 출력 헤드
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_classes)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _pad_features(self, x: torch.Tensor) -> torch.Tensor:
        """피처를 최대 차원으로 패딩"""
        current_dim = x.size(-1)
        if current_dim < self.max_input_dim:
            padding = torch.zeros(x.size(0), x.size(1), self.max_input_dim - current_dim, device=x.device)
            x = torch.cat([x, padding], dim=-1)
        return x
    
    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, 
                test_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            context_x: [batch_size, context_len, feature_dim]
            context_y: [batch_size, context_len]
            test_x: [batch_size, test_len, feature_dim]
        Returns:
            logits: [batch_size, test_len, num_classes]
        """
        batch_size = context_x.size(0)
        context_len = context_x.size(1)
        test_len = test_x.size(1)
        device = context_x.device
        
        # 피처 정규화 및 패딩
        context_x = self.feature_norm(self._pad_features(context_x))
        test_x = self.feature_norm(self._pad_features(test_x))
        
        # 컨텍스트 임베딩
        context_x_emb = self.feature_embed(context_x)
        context_y_emb = self.label_embed(context_y)
        context_emb = context_x_emb + context_y_emb
        
        # 테스트 임베딩
        test_emb = self.feature_embed(test_x)
        
        # 위치 임베딩 추가
        context_emb = context_emb + self.position_embed[:context_len].unsqueeze(0)
        test_emb = test_emb + self.position_embed[context_len:context_len+test_len].unsqueeze(0)
        
        # 타입 임베딩 추가
        context_emb = context_emb + self.type_embed[0].unsqueeze(0).unsqueeze(0)
        test_emb = test_emb + self.type_embed[1].unsqueeze(0).unsqueeze(0)
        
        # 전체 시퀀스 구성
        full_sequence = torch.cat([context_emb, test_emb], dim=1)
        
        # Transformer 인코더 통과
        encoded = self.transformer(full_sequence)
        
        # 테스트 부분 추출
        test_encoded = encoded[:, context_len:context_len+test_len]
        
        # 크로스 어텐션 적용
        context_encoded = encoded[:, :context_len]
        attended = self.cross_attention(test_encoded, context_encoded, context_encoded)
        
        # 최종 출력
        logits = self.output_head(attended)
        
        return logits