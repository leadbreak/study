import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from tqdm import tqdm
import time

from torchsummary import summary

from model.fcmae import convnextv2_fcmae_tiny
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
    
model = convnextv2_fcmae_tiny()

# Transforms 정의하기
train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

data_dir = '../../data/sports/'
batch_size = 800
train_path = data_dir

# dataset load
train_data = ImageFolder(train_path, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)

len(train_loader)
device = 'cuda:0'
model.to(device)

model_path = '../../model/convnext/fcmae.pt'

epochs = 500
optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))

warmup_steps = int(len(train_loader)*(epochs)*0.1)
train_steps = len(train_loader)*(epochs)
scheduler = CosineWarmupScheduler(optimizer, 
                                num_warmup_steps=50, 
                                num_training_steps=500,
                                num_cycles=0.5,
                                min_lr=1e-7)

training_time = 0
losses = []
val_losses = []
lrs = []
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
    
    for _, data in pbar:
        optimizer.zero_grad()
        
        samples= data[0].to(device)
        loss, _, _ = model(samples, mask_ratio=0.6)
        
        loss.backward()
        optimizer.step()
            
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)        
    
    # 모델 저장 로직 조정
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        if epoch > (epochs // 2) :
            torch.save(model.state_dict(), model_path)
            model_saved_text = ' - model saved!'
        else :
            model_saved_text = ' - model save pass'
    else:
        model_saved_text = ''
    
    epoch_duration = time.time() - start_time
    training_time += epoch_duration
    
    text = f'\tLoss: {epoch_loss:,.4f}, LR: {lr}, Duration: {epoch_duration:.2f} sec{model_saved_text}'
    print(text)

    # 에폭마다 스케줄러 업데이트
    scheduler.step()
