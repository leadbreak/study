import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import time       
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import swin_v1 as swin
from timm.data import Mixup
from torch.nn.utils import clip_grad_norm_
import transformers

import click

def load_data(img_size:int,
              train_option:str,
              data_dir:str,
              batch_size:int):
        
    # Transforms 정의하기
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8,1), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.9, scale=(0.02, 0.33)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if train_option == 'total':
        train_path = valid_path = test_path = data_dir        
    elif train_option == 'holdout':
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
    
    return train_loader, valid_loader, test_loader

@click.command()
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
def main(data_dir:str='../data/sports',
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
    
    model = swin.SwinTransformer(img_size=img_size, 
                            patch_size=patch_size,
                            window_size=window_size,
                            in_chans=in_chans, 
                            num_classes=num_classes, 
                            mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            **args).to(device)
    
    train_loader, valid_loader, test_loader = load_data(img_size=img_size, train_option=train_option, data_dir=data_dir, batch_size=batch_size)
    
    mixup_fn = Mixup(mixup_alpha=mixup_alpha, 
                    cutmix_alpha=cutmix_alpha, 
                    prob=mixup_prob, 
                    switch_prob=0.5, 
                    mode='batch',
                    label_smoothing=mixup_label_smoothing,
                    num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        model.train()
        start_time = time.time()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
        
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
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 옵티마이저 스텝 및 스케일러 업데이트
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            lr = optimizer.param_groups[0]["lr"]
            lrs.append(lr)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
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
            
        elif train_option == 'holdout':
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
            
            # 모델 저장
            if val_loss < best_loss:
                best_loss = val_loss
                vit_save = True
                torch.save(model.state_dict(), model_path)

            epoch_duration = time.time() - start_time
            training_time += epoch_duration
            
            text = f'\tLoss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
            
            if vit_save:
                text += f' - model saved!'
                vit_save = False

        click.echo(text)
            
    text = f"Epoch 당 평균 소요시간 : {training_time / epochs:.2f}초"
    styled_text = click.style(text, fg='cyan', bold=True)            
    click.echo(styled_text)
        
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
    click.echo(performance_metrics)

if __name__ == '__main__':
    main()