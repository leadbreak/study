import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T

import numpy as np
import time       
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from timm.data import Mixup
from torch.nn.utils import clip_grad_norm_
import transformers
import timm

import click

# MaskGenerator 클래스 정의
class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return torch.tensor(mask, dtype=torch.float32)

# SimMIMTransform 클래스 정의
class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config['IMG_SIZE'], scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ])
        self.mask_generator = MaskGenerator(
            input_size=config['IMG_SIZE'],
            mask_patch_size=config['MASK_PATCH_SIZE'],
            model_patch_size=config['MODEL_PATCH_SIZE'],
            mask_ratio=config['MASK_RATIO'],
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        return img, mask

# collate_fn 함수 정의
def collate_fn(batch):
    batch_images, batch_masks = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_masks = torch.stack(batch_masks)
    return batch_images, batch_masks

# DataLoader 구성
def build_loader_simmim(config):
    transform = SimMIMTransform(config)
    dataset = ImageFolder(config['DATA_PATH'], transform)
    dataloader = DataLoader(dataset, config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True, drop_last=True, collate_fn=collate_fn)
    return dataloader


@click.command()
@click.option('--define_model', default='self')
@click.option('--train_option', default='holdout')
@click.option('--data_dir', default='../../data/sports')
@click.option('--model_path', default='../../models/swin/model.pth')
@click.option('--epochs', default=10)
@click.option('--batch_size', default=512)
@click.option('--label_smoothing', default=0.)
@click.option('--mixup_label_smoothing', default=0.1)
@click.option('--learning_rate', default=1e-3)
@click.option('--device', default='cuda:0')
@click.option('--img_size', default=224)
@click.option('--patch_size', default=4)
@click.option('--window_size', default=7)
@click.option('--num_classes', default=100)
@click.option('--drop_rate', default=0.)
@click.option('--attn_drop_rate', default=0.)
@click.option('--mlp_ratio', default=4.)
@click.option('--model_type', default='tiny')
@click.option('--weight_decay', default=5e-3)
@click.option('--mixup_alpha', default=0.7)
@click.option('--cutmix_alpha', default=0.7)
@click.option('--mixup_prob', default=1.0)
def main(define_model:str='self',
         data_dir:str='../data/sports',
         train_option:str='total',
         model_path:str=None,
         epochs:int=10,
         batch_size:int=512,
         label_smoothing:float=0.,
         learning_rate:float=1e-3,
         device:str='cuda:0',
         img_size:int=224,
         patch_size:int=4,
         in_chans:int=3,
         num_classes:int=100,
         window_size:int=7,
         mlp_ratio:float=4.0,
         drop_rate:float=0.,
         attn_drop_rate:float=0.,
         model_type:str='tiny',
         weight_decay:float=5e-3,
         mixup_label_smoothing:float=0.1,
         mixup_alpha:float=0.3,
         cutmix_alpha:float=0.3,
         mixup_prob:float=0.7,
         ):
    
    if define_model == 'timm':
        model_name = f'swin_{model_type}_patch{patch_size}_window{window_size}_{img_size}.ms_in22k'
        
    elif define_model == 'swin_v1':
        args = {}
        if model_type == 'small':
            args['embed_dim'] = 96
            args['heads'] = [3,6,12,24]
            args['depths'] = [2,2,18,2]
            args['drop_path_rate'] = 0.3
        elif model_type == 'base':
            args['embed_dim'] = 128
            args['heads'] = [4,8,16,32]
            args['depths'] = [2,2,18,2]
            args['drop_path_rate'] = 0.5
        elif model_type == 'large':
            args['embed_dim'] = 96
            args['heads'] = [6,12,24,68]
            args['depths'] = [2,2,18,2]
            args['drop_path_rate'] = 0.5
        else:
            args['embed_dim'] = 96
            args['heads'] = [3,6,12,24]
            args['depths'] = [2,2,6,2]
            args['drop_path_rate'] = 0.2
    
    elif define_model == 'swin_v2':
        args = {}
        if model_type == 'small':
            args['embed_dim'] = 96
            args['heads'] = [3,6,12,24]
            args['depths'] = [2,2,18,2]
            args['drop_path_rate'] = 0.3
        elif model_type == 'base':
            args['embed_dim'] = 128
            args['heads'] = [4,8,16,32]
            args['depths'] = [2,2,18,2]
            args['drop_path_rate'] = 0.5
        elif model_type == 'large':
            args['embed_dim'] = 96
            args['heads'] = [6,12,24,68]
            args['depths'] = [2,2,18,2]
            args['drop_path_rate'] = 0.5
        else:
            args['embed_dim'] = 96
            args['heads'] = [3,6,12,24]
            args['depths'] = [2,2,6,2]
            args['drop_path_rate'] = 0.2
        
    
    # 파일명이 지정되지 않으면 시간으로
    if model_path is None:
        current_time = datetime.now()
        model_path = current_time.strftime("%y%m%d_%H%M") + ".pth"
        
    if define_model == 'timm':
        model = timm.create_model(model_name=model_name,
                                  pretrained=False,
                                  num_classes=num_classes)
    elif define_model == 'swin_v1':
        import swin_v1 as swin
        model = swin.SwinTransformer(img_size=img_size, 
                                    patch_size=patch_size,
                                    window_size=window_size,
                                    in_chans=in_chans, 
                                    num_classes=num_classes, 
                                    mlp_ratio=mlp_ratio,
                                    drop_rate=drop_rate,
                                    attn_drop_rate=attn_drop_rate,
                                    **args)
    elif define_model == 'swin_v2':
        import swin_v2 as swin
        model = swin.SwinTransformerV2(img_size=img_size, 
                                       patch_size=patch_size,
                                       window_size=window_size,
                                       in_chans=in_chans, 
                                       num_classes=num_classes, 
                                       mlp_ratio=mlp_ratio,
                                       drop_rate=drop_rate,
                                       attn_drop_rate=attn_drop_rate,
                                       **args)   
    
    model.to(device)
    
    # Config 설정
    config = {
        'DATA_PATH': '/path/to/your/dataset',
        'IMG_SIZE': 224,
        'MASK_PATCH_SIZE': 32,
        'MODEL_PATCH_SIZE': 4,
        'MASK_RATIO': 0.6,
        'BATCH_SIZE': 32,
        'NUM_WORKERS': 4
    }
    
    dataloader = build_loader_simmim(config)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = int(len(dataloader)*epochs*0.1)
    train_steps = len(dataloader)*epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                            num_warmup_steps=warmup_steps, 
                                                            num_training_steps=train_steps,
                                                            num_cycles=0.5)

    training_time = 0
    losses = []
    lrs = []
    best_loss = float('inf')

    # GradScaler 초기화
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}")
        
        for _, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # AutoCast 적용
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # 스케일링된 그라디언트 계산
            scaler.scale(loss).backward()

            # 그라디언트 클리핑 전에 스케일링 제거
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 옵티마이저 스텝 및 스케일러 업데이트
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            lr = optimizer.param_groups[0]["lr"]
            lrs.append(lr)
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        
        if train_option == 'total':
            # 모델 저장
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                vit_save = True
                torch.save(model.state_dict(), model_path)
                
            epoch_duration = time.time() - start_time
            training_time += epoch_duration
            
            text = f'\tLoss: {epoch_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
            
            if vit_save:
                text += f' - model saved!'
                vit_save = False    

        click.echo(text)
            
    text = f"Epoch 당 평균 소요시간 : {training_time / epochs:.2f}초"
    styled_text = click.style(text, fg='cyan', bold=True)            
    click.echo(styled_text)

if __name__ == '__main__':
    main()