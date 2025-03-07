import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

import time       
from datetime import datetime
from tqdm import tqdm

import vit_better as vit_custom
import click

def load_data(img_size:int,
              data_dir:str,
              batch_size:int):
        
    # Transforms 정의하기
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8,1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.9, scale=(0.02, 0.33)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dataset load
    train_data = ImageFolder(data_dir+'/train', transform=train_transform)
    valid_data = ImageFolder(data_dir+'/valid', transform=test_transform)
    test_data = ImageFolder(data_dir+'/test', transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

@click.command()
@click.option('--data_dir', default='../data/sports')
@click.option('--model_path', default=None)
@click.option('--epochs', default=3)
@click.option('--batch_size', default=64)
@click.option('--label_smoothing', default=0.1)
@click.option('--learning_rate', default=1e-3)
@click.option('--device', default='cuda:0')
@click.option('--img_size', default=224)
@click.option('--patch_size', default=16)
@click.option('--num_classes', default=100)
@click.option('--dropout', default=0.1)
@click.option('--embed_dim', default=768)
@click.option('--num_layers', default=12)
@click.option('--num_heads', default=12)
@click.option('--mlp_ratio', default=4.)
@click.option('--estimate_params', default=True)
@click.option('--fused_attention', default=True)
def main(data_dir:str='../data/sports',
         model_path:str=None,
         epochs:int=10,
         batch_size:int=64,
         label_smoothing:float=0.1,
         learning_rate:float=1e-3,
         device:str='cuda:0',
         img_size:int=224,
         patch_size:int=18,
         num_classes:int=100,
         dropout:float=0.1,
         embed_dim:int=768,
         num_layers:int=12,
         num_heads:int=12,
         mlp_ratio:float=4.0,
         estimate_params:bool=True,
         fused_attention:bool=True,
         ):
    
    # 파일명이 지정되지 않으면 시간으로
    if model_path is None:
        current_time = datetime.now()
        model_path = current_time.strftime("%y%m%d_%H%M") + ".pth"
    
    model = vit_custom.VisionTransformer(img_size=img_size, 
                            patch_size=patch_size, 
                            num_classes=num_classes, 
                            dropout=dropout,
                            embed_dim=embed_dim,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            estimate_params=estimate_params,
                            fused_attention=fused_attention).to(device)
    
    train_loader, valid_loader, test_loader = load_data(img_size=img_size, data_dir=data_dir, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)*10, gamma=0.5)

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
        
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]
            lrs.append(lr)
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
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            model_save = True
            torch.save(model.state_dict(), model_path)

        epoch_duration = time.time() - start_time
        training_time += epoch_duration
        
        text = f'\tLoss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec'
        
        if model_save:
            text += f' - model saved!'
            model_save = False

        click.echo(text)
            
    text = f"Epoch 당 평균 소요시간 : {training_time / epochs:.2f}초"
    styled_text = click.style(text, fg='cyan', bold=True)            
    click.echo(styled_text)

if __name__ == '__main__':
    main()