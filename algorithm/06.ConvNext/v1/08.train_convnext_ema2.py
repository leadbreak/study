import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time

from timm.data import Mixup
from timm.utils import ModelEmaV3
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import transformers

from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from convnext import load_convNext
from torchsummary import summary
import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr=1e-6, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.num_warmup_steps:
                # Linear warmup
                lr = (base_lr - self.min_lr) * self.last_epoch / max(1, self.num_warmup_steps) + self.min_lr
            else:
                # Cosine annealing
                progress = (self.last_epoch - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            lrs.append(lr)
        return lrs

model = load_convNext()

# 총 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())

# 학습 가능한 파라미터 수 계산
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('='*80)
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}\n")
print('='*80)

model_summary = summary(model.cuda(), (3, 224, 224))
print(model_summary)


# Transforms 정의하기
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6,1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=1., scale=(0.02, 0.33)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = '../../data/sports'
batch_size = 800

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

device = 'cuda:3'
max_norm = 3.0 

model.to(device)

model_ema = None
ema_active = True
if ema_active:
    ema_decay = 0.9999
    model_ema = ModelEmaV3(
        model,
        decay=ema_decay,
    )
    print(f"Using EMA with decay = {ema_decay}")

model_path = ''

mixup = True
if mixup :
    mixup_fn = Mixup(mixup_alpha=.8, 
                    cutmix_alpha=1., 
                    prob=1., 
                    switch_prob=0.5, 
                    mode='batch',
                    label_smoothing=.1,
                    num_classes=100)
    
    criterion = SoftTargetCrossEntropy()
else :
    criterion = LabelSmoothingCrossEntropy(.1)

criterion2 = nn.CrossEntropyLoss()
epochs = 500

optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.05)
warmup_steps = int(len(train_loader)*(epochs)*0.1)
train_steps = len(train_loader)*(epochs)
scheduler = CosineWarmupScheduler(optimizer, 
                                num_warmup_steps=warmup_steps, 
                                num_training_steps=train_steps,
                                num_cycles=0.5,
                                min_lr=1e-6)
# scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
#                                                         num_warmup_steps=warmup_steps, 
#                                                         num_training_steps=train_steps,
#                                                         num_cycles=0.5)

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
            
            # EMA 모델 업데이트
            if model_ema is not None:
                model_ema.update(model)
                
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
                loss = criterion2(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(valid_loader)
        val_losses.append(val_loss)
        
        # 모델 저장
        total_loss = val_loss + epoch_loss
        if total_loss < best_loss:
            best_loss = total_loss
            model_save = False
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

