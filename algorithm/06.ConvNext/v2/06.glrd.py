'''

'''

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import time

from timm.data import Mixup
from timm.utils import ModelEmaV3
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from collections import OrderedDict
from model.convnextv2 import load_convNext
import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler

print("이전 학습 대기 중...")
time.sleep(101*1000)

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
        
# checkpoint_model = convnextv2_fcmae_tiny()
model = load_convNext(droppath=0.2)

def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = '.'.join(k.split('.')[1:]) # remove encoder in the name
        if k.endswith('kernel'):
            k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3: # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).\
                    reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2: # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).\
                    reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2) # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

pretrain_path = '../../model/convnext/fcmae.pt'
checkpoint_model = torch.load(pretrain_path, map_location='cpu')

state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from head of pretrained checkpoint")
        del checkpoint_model[k]

# remove decoder weights
checkpoint_model_keys = list(checkpoint_model.keys())
for k in checkpoint_model_keys:
    if 'decoder' in k or 'mask_token'in k or \
        'proj' in k or 'pred' in k:
        print(f"Removing key {k} from decoder of pretrained checkpoint")
        del checkpoint_model[k]

checkpoint_model = remap_checkpoint_keys(checkpoint_model)
load_state_dict(model, checkpoint_model, prefix='')

# 총 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())

# 학습 가능한 파라미터 수 계산
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('='*80)
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}\n")
print('='*80)

# model_summary = summary(model.cuda(), (3, 224, 224))

# Transforms 정의하기
train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(interpolation=F.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(224, scale=(0.6,1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = '../../data/sports'
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

device = 'cuda:4'
max_norm = 3.0 

model.to(device)

model_ema = None
ema_active = True
if ema_active:
    ema_decay = 0.998
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
    
criterion = nn.CrossEntropyLoss(label_smoothing=0.)

# LLRD
def LLRD_ConvNeXt(model, depths=[3,3,9,3], weight_decay=0.05, lr=8e-3, scale=0.98):
    
    stage = 0
    layer_names = []
    param_groups = {}
    for depth in depths:
        if stage == 0:
            layer_names.append(f'downsample_layers.{stage}.stem_conv.weight')
            layer_names.append(f'downsample_layers.{stage}.stem_conv.bias')
            layer_names.append(f'downsample_layers.{stage}.stem_ln.weight')
            layer_names.append(f'downsample_layers.{stage}.stem_ln.bias')
        else :
            layer_names.append(f'downsample_layers.{stage}.ds_ln.weight')
            layer_names.append(f'downsample_layers.{stage}.ds_ln.bias')
            layer_names.append(f'downsample_layers.{stage}.ds_conv.weight')
            layer_names.append(f'downsample_layers.{stage}.ds_conv.bias')        
        for i in range(depth):
            layer_names.append(f'stages.{stage}.{i}.dwconv.weight')
            layer_names.append(f'stages.{stage}.{i}.dwconv.bias')
            layer_names.append(f'stages.{stage}.{i}.layernorm.weight')
            layer_names.append(f'stages.{stage}.{i}.layernorm.bias')
            layer_names.append(f'stages.{stage}.{i}.pwconv1.weight')
            layer_names.append(f'stages.{stage}.{i}.pwconv1.bias')
            layer_names.append(f'stages.{stage}.{i}.grn.gamma')
            layer_names.append(f'stages.{stage}.{i}.grn.beta')            
            layer_names.append(f'stages.{stage}.{i}.pwconv2.weight')
            layer_names.append(f'stages.{stage}.{i}.pwconv2.bias')
        stage += 1
    
    layer_names.append('layernorm.weight')
    layer_names.append('layernorm.bias')
    layer_names.append('fc.weight')
    layer_names.append('fc.bias')
    
    # Layer Learning Rate Decay
    for name, param in model.named_parameters():
        total_depths = sum(depths)
        if name.startswith("downsample_layers"):
            stage_id = int(name.split('.')[1])
            layer_id = sum(depths[:stage_id]) + 1
            param_groups[name] = {'lr':lr*(scale**((total_depths-layer_id)//3+1)),
                                  'weight_decay':0.}
        
        elif name.startswith("stages"):
            stage_id = int(name.split('.')[1])
            block_id = int(name.split('.')[2])
            layer_id = sum(depths[:stage_id]) + block_id + 1
            if len(param.shape) == 1 or name.endswith(".bias") or name.endswith(".gamma") or name.endswith(".beta"):
                param_groups[name] = {'lr':lr*(scale**((total_depths-layer_id)//3+1)),
                                      'weight_decay':0.}
            else :
                param_groups[name] = {'lr':lr*(scale**((total_depths-layer_id)//3+1)),
                                      'weight_decay':weight_decay}       
        else : # head
            if len(param.shape) == 1 or name.endswith(".bias"):
                param_groups[name] = {'lr':lr,
                                      'weight_decay':0.}
            else :
                param_groups[name] = {'lr':lr,
                                      'weight_decay':weight_decay}    
    return layer_names, param_groups

layer_names, param_groups = LLRD_ConvNeXt(model)
groups = [{'params': param,
            'lr' : param_groups[name]['lr'],
            'weight_decay': param_groups[name]['weight_decay']} for name, param in model.named_parameters()]

epochs = 1000

optimizer = optim.AdamW(groups)
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
model_save = False

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
                
            loss.backward()
            # 그라디언트 클리핑 적용
            clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            # EMA 모델 업데이트, 필요한 경우
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
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(valid_loader)
        val_losses.append(val_loss)
        
        # 모델 저장 조건 수정
        total_loss = val_loss + epoch_loss
        if total_loss < best_loss:
            best_loss = total_loss
            # torch.save(model.state_dict(), model_path)
            model_save = True
            save_text = ' - model saved!'
        else:
            save_text = ''

        epoch_duration = time.time() - start_time
        training_time += epoch_duration
        
        text = f'\tLoss: {epoch_loss:.4f}, Val_Loss: {val_loss:.4f}, Total Mean Loss: {total_loss/2:.4f}, LR: {lr}, Duration: {epoch_duration:.2f} sec{save_text}'
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
