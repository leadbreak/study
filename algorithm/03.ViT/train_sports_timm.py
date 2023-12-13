import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

img_size = 224
num_classes = 100
batch_size = 512
num_workers = 8

model_name = 'vit_base_patch16_224'
pretrained = False

device_gpu = 'cuda:3'

label_smoothing = 0.1
learning_rate = 0.001
epochs = 1000

model_path = 'test_model.pth'  # 모델 저장 경로

# 데이터셋 경로 설정
data_dir = './data/sports'  # Tiny ImageNet 데이터셋이 저장된 경로

# Transforms 정의하기
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8,1), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.9, scale=(0.02, 0.33)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# dataset load
train_data = ImageFolder('./data/sports/train', transform=train_transform)
valid_data = ImageFolder('./data/sports/valid', transform=test_transform)
test_data = ImageFolder('./data/sports/test', transform=test_transform)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = timm.create_model(model_name=model_name, 
                          pretrained=pretrained, 
                          num_classes=num_classes)

device = torch.device(device_gpu if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
step_size = len(trainloader)*50
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, min_lr=1e-5)

from tqdm import tqdm

training_time = 0
losses = []
val_losses = []
lrs = []
best_val_loss = float('inf')

model_save = False

# GradScaler 초기화
scaler = GradScaler()

for epoch in range(epochs):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}")
    
    for i, data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        # AutoCast 적용
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Scaled Backward & Optimizer Step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        running_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
    epoch_loss = running_loss / len(trainloader)
    losses.append(epoch_loss)

    # 검증 손실 계산
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(testloader)
    val_losses.append(val_loss)

    # 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_save = True
        torch.save(model.state_dict(), model_path)

    epoch_duration = time.time() - start_time
    training_time += epoch_duration
    if model_save:
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec - model saved!')
        model_save = False
    else :
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Val Loss: {val_loss}, LR: {lr}, Duration: {epoch_duration:.2f} sec')

torch.save(model.state_dict(), './last_test_sports.pth')

# 예측 수행 및 레이블 저장
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in testloader:
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
print(performance_metrics)