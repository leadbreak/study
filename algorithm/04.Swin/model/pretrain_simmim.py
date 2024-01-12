import numpy as np
import yaml
from box import Box

import torch
import torch.nn as nn
import torch.optim as optim

import simmim
from swin_v2 import SwinTransformerV2

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time

from timm.data import Mixup
import transformers
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


simmim_config = yaml.load(open('config/pretrain.yaml'), Loader=yaml.FullLoader)

encoder_config = {'img_size':simmim_config['DATA']['IMG_SIZE'], 
                'patch_size':simmim_config['MODEL']['SWIN']['PATCH_SIZE'], 
                'in_chans':3, 
                'num_classes':100,
                'embed_dim':simmim_config['MODEL']['SWIN']['EMBED_DIM'], 
                'depths':simmim_config['MODEL']['SWIN']['DEPTHS'], 
                'num_heads':simmim_config['MODEL']['SWIN']['NUM_HEADS'],           
                'window_size':simmim_config['MODEL']['SWIN']['WINDOW_SIZE'], 
                'mlp_ratio':4., 
                'qkv_bias':True, 
                'qk_scale':None,
                'drop_rate':0., 
                'attn_drop_rate':0., 
                'drop_path_rate':simmim_config['MODEL']['DROP_PATH_RATE'],
                'norm_layer':nn.LayerNorm, 
                'patch_norm':True, 
                'pretrained_window_sizes':[0,0,0,0],
                'ape':True}

encoder_stride = 32
in_chans = encoder_config['in_chans']
patch_size = encoder_config['patch_size']

encoder = simmim.SwinTransformerV2ForSimMIM(**encoder_config)

model = simmim.SimMIM( encoder=encoder, 
                       encoder_stride=encoder_stride, 
                       in_chans=in_chans, 
                       patch_size=patch_size)

simmim_config = Box(simmim_config)
dataloader = simmim.build_loader_simmim(simmim_config)

base_lr = float(simmim_config.TRAIN.BASE_LR)
weight_decay = simmim_config.TRAIN.WEIGHT_DECAY
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
warmup_epochs = simmim_config.TRAIN.WARMUP_EPOCHS
train_epochs = simmim_config.TRAIN.EPOCHS

multisteps = simmim_config.TRAIN.LR_SCHEDULER.MULTISTEPS
gamma = simmim_config.TRAIN.LR_SCHEDULER.GAMMA
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=multisteps, gamma=gamma)

# LambdaLR 스케줄러 설정
lambda1 = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1 # Warmup을 위한 Lambda 함수 정의
scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# MultiStepLR 스케줄러 설정
scheduler_multistep = optim.lr_scheduler.MultiStepLR(optimizer, milestones=multisteps, gamma=gamma)

device = 'cuda:3'
model.to(device)

model_save = True
model_path = '../../models/swin2/simmim.pth'

training_time = 0
losses = []
val_losses = []
lrs = []
best_loss = float('inf')

# GradScaler 초기화
scaler = GradScaler()

for epoch in range(train_epochs):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
    
    for _, data in pbar:
        image, mask = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        # AutoCast 적용
        with autocast():
            loss = model(image, mask)
            
        # 스케일링된 그라디언트 계산
        scaler.scale(loss).backward()

        # 그라디언트 클리핑 전에 스케일링 제거
        scaler.unscale_(optimizer)
        if simmim_config.TRAIN.CLIP_GRAD:
            clip_grad_norm_(model.parameters(), max_norm=simmim_config.TRAIN.CLIP_GRAD)
        else:
            clip_grad_norm_(model.parameters())

        # 옵티마이저 스텝 및 스케일러 업데이트
        scaler.step(optimizer)
        scaler.update()
        if epoch <= warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_multistep.step()
        # scheduler.step()
            
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    losses.append(epoch_loss)

    # 모델 저장
    if epoch_loss < best_loss:
        
        best_loss = epoch_loss
        vit_save = model_save
        if vit_save:
            torch.save(model.state_dict(), model_path)
        
    epoch_duration = time.time() - start_time
    training_time += epoch_duration
    
    text = f'\tLoss: {epoch_loss:.4f}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
    
    if vit_save:
        text += f' - model saved!'
        vit_save = False    
        
    print(text)

swin = SwinTransformerV2(pretrained_window_sizes=[7,7,7,7], ape=True)
swin.load_state_dict(model.encoder.state_dict(), strict=False)

# Transforms 정의하기
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.9, scale=(0.02, 0.33)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = '../../data/sports'
batch_size = 960

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

torch.cuda.empty_cache()

max_norm = 5.0
swin.to(device)

mixup_fn = Mixup(mixup_alpha=1., 
                cutmix_alpha=1., 
                prob=1., 
                switch_prob=0.5, 
                mode='batch',
                label_smoothing=.1,
                num_classes=100)

epochs = 500

criterion = nn.CrossEntropyLoss(label_smoothing=0.)
optimizer = optim.AdamW(swin.parameters(), lr=1e-3, weight_decay=5e-3)
warmup_steps = int(len(train_loader)*epochs*0.1)
train_steps = len(train_loader)*epochs
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

for epoch in range(epochs):
    swin.train()
    start_time = time.time()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
    
    for _, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = mixup_fn(inputs, labels)
        optimizer.zero_grad()

        # AutoCast 적용
        with autocast():
            outputs = swin(inputs)
            loss = criterion(outputs, labels)
            
        # 스케일링된 그라디언트 계산
        scaler.scale(loss).backward()

        # 그라디언트 클리핑 전에 스케일링 제거
        scaler.unscale_(optimizer)
        clip_grad_norm_(swin.parameters(), max_norm=max_norm)

        # 옵티마이저 스텝 및 스케일러 업데이트
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
            
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)        

    swin.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = swin(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    val_loss /= len(valid_loader)
    val_losses.append(val_loss)
    
    # 모델 저장
    if val_loss < best_loss:
        best_loss = val_loss
        # vit_save = True
        # if vit_save:
        #     torch.save(swin.state_dict(), )

    epoch_duration = time.time() - start_time
    training_time += epoch_duration
    
    text = f'\tLoss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
    
    # if vit_save:
    #     text += f' - swin saved!'
    #     vit_save = False

    print(text)
        
text = f"Epoch 당 평균 소요시간 : {training_time / epochs:.2f}초"      
print(text)

# 예측 수행 및 레이블 저장
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = swin(images)
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