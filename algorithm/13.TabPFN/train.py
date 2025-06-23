import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")

from tabPFN import TabPFNV2


class FocalLoss(nn.Module):
    """클래스 불균형 처리를 위한 Focal Loss"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LearningRateScheduler:
    """커스텀 학습률 스케줄러"""
    
    def __init__(self, optimizer, warmup_steps: int = 1000, max_lr: float = 1e-3):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * (self.step_count - self.warmup_steps) / (10000 - self.warmup_steps)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class MetricTracker:
    """메트릭 추적기"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.count = 0
    
    def update(self, loss: float, accuracy: float):
        self.total_loss += loss
        self.total_accuracy += accuracy
        self.count += 1
    
    def get_averages(self) -> Tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        return self.total_loss / self.count, self.total_accuracy / self.count

# ----------------------------
# TabPFN V2 Learner
# ----------------------------

class TabPFNV2Learner:
    """TabPFN V2 학습 관리자"""
    
    def __init__(self, model: TabPFNV2, device: torch.device, 
                 lr: float = 1e-4, weight_decay: float = 1e-5,
                 use_focal_loss: bool = True):
        self.model = model.to(device)
        self.device = device
        
        # 최적화기 설정
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 손실 함수
        self.criterion = FocalLoss() if use_focal_loss else nn.CrossEntropyLoss()
        
        # 스케줄러
        self.scheduler = LearningRateScheduler(self.optimizer, warmup_steps=1000, max_lr=lr)
        
        # 메트릭 추적기
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """단일 훈련 스텝"""
        self.model.train()
        context_x, context_y, test_x, test_y = batch
        
        # 배치 차원 처리
        context_x = context_x.squeeze(0).unsqueeze(0)  # [1, context_len, feature_dim]
        context_y = context_y.squeeze(0).unsqueeze(0)  # [1, context_len]
        test_x = test_x.squeeze(0).unsqueeze(0)        # [1, test_len, feature_dim]
        test_y = test_y.squeeze(0)                     # [test_len]
        
        # 디바이스 이동
        context_x = context_x.to(self.device)
        context_y = context_y.to(self.device)
        test_x = test_x.to(self.device)
        test_y = test_y.to(self.device)
        
        # Forward pass
        logits = self.model(context_x, context_y, test_x)
        logits = logits.squeeze(0)  # [test_len, num_classes]
        
        # 손실 계산
        loss = self.criterion(logits, test_y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # 정확도 계산
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == test_y).float().mean().item()
        
        return loss.item(), accuracy
    
    def validate_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        """검증 스텝"""
        self.model.eval()
        with torch.no_grad():
            context_x, context_y, test_x, test_y = batch
            
            # 배치 차원 처리
            context_x = context_x.squeeze(0).unsqueeze(0)
            context_y = context_y.squeeze(0).unsqueeze(0)
            test_x = test_x.squeeze(0).unsqueeze(0)
            test_y = test_y.squeeze(0)
            
            # 디바이스 이동
            context_x = context_x.to(self.device)
            context_y = context_y.to(self.device)
            test_x = test_x.to(self.device)
            test_y = test_y.to(self.device)
            
            # Forward pass
            logits = self.model(context_x, context_y, test_x)
            logits = logits.squeeze(0)
            
            # 손실 및 정확도 계산
            loss = self.criterion(logits, test_y)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == test_y).float().mean().item()
            
            return loss.item(), accuracy
    
    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """전체 에포크 훈련"""
        self.train_metrics.reset()
        
        for batch in train_loader:
            loss, accuracy = self.train_step(batch)
            self.train_metrics.update(loss, accuracy)
        
        train_loss, train_acc = self.train_metrics.get_averages()
        results = {'train_loss': train_loss, 'train_accuracy': train_acc}
        
        if val_loader is not None:
            self.val_metrics.reset()
            for batch in val_loader:
                loss, accuracy = self.validate_step(batch)
                self.val_metrics.update(loss, accuracy)
            
            val_loss, val_acc = self.val_metrics.get_averages()
            results.update({'val_loss': val_loss, 'val_accuracy': val_acc})
        
        return results