import torch
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # Corrected import
from transformers import get_cosine_schedule_with_warmup  # Import scheduler

from model.convnextv2 import ConvNeXtV2
from model.convnextv2_moe import ConvNeXtV2_MoE
from model.convnextv2_moe_grn import ConvNeXtV2_MoE_GRN

# Hyperparameters

## Model parameters
num_classes = 10  # CIFAR-10 classes

## Training parameters
batch_size = 256
epochs = 100
lambda_cov = 0.1  # Coefficient for auxiliary loss (if any)

# Initialize models
convnext = ConvNeXtV2(num_classes=num_classes)
convnext_moe = ConvNeXtV2_MoE(num_classes=num_classes)
convnext_moe_grn = ConvNeXtV2_MoE_GRN(num_classes=num_classes)

from torchinfo import summary

summary(convnext, input_size=(1, 3, 32, 32), depth=3, col_names=["input_size", "output_size", "num_params"])
summary(convnext_moe, input_size=(1, 3, 32, 32), col_names=["input_size", "output_size", "num_params"])
summary(convnext_moe_grn, input_size=(1, 3, 32, 32), depth=3, col_names=["input_size", "output_size", "num_params"])

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train(model, train_loader, optimizer, scheduler, criterion, epochs=1):
    model.to(device)
    model.train()
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():  # Use autocast for mixed precision
                if isinstance(model, (ConvNeXtV2_MoE, ConvNeXtV2_MoE_GRN)):
                    outputs, l_aux = model(images)
                    loss = criterion(outputs, labels) + lambda_cov * l_aux
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Scale loss before backward pass
            scaler.step(optimizer)         # Optimizer step
            scaler.update()                # Update the scaler

            scheduler.step()  # Update the learning rate

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

# Test function (unchanged)
def test(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            if isinstance(model, (ConvNeXtV2_MoE, ConvNeXtV2_MoE_GRN)):
                outputs, _ = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data augmentation and loading
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
])

transform_test = transforms.Compose([
    transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./cifar10_data/', train=False, transform=transform_test, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define criterion
criterion = nn.CrossEntropyLoss()

# Function to calculate total training steps
def get_total_steps(loader, epochs):
    return len(loader) * epochs

# Training ConvNeXtV2
optimizer_convnext = optim.AdamW(convnext.parameters(), lr=0.001)
total_steps_convnext = get_total_steps(train_loader, epochs)
warmup_steps_convnext = int(0.1 * total_steps_convnext)  # 10% of total steps
scheduler_convnext = get_cosine_schedule_with_warmup(optimizer_convnext, num_warmup_steps=warmup_steps_convnext, num_training_steps=total_steps_convnext)

train(convnext, train_loader, optimizer_convnext, scheduler_convnext, criterion, epochs=epochs)
test(convnext, test_loader)

# Training ConvNeXtV2_MoE
optimizer_moe = optim.AdamW(convnext_moe.parameters(), lr=0.001)
total_steps_moe = get_total_steps(train_loader, epochs)
warmup_steps_moe = int(0.1 * total_steps_moe)  # 10% of total steps
scheduler_moe = get_cosine_schedule_with_warmup(optimizer_moe, num_warmup_steps=warmup_steps_moe, num_training_steps=total_steps_moe)

train(convnext_moe, train_loader, optimizer_moe, scheduler_moe, criterion, epochs=epochs)
test(convnext_moe, test_loader)

# Training ConvNeXtV2_MoE_GRN
optimizer_moe_grn = optim.AdamW(convnext_moe_grn.parameters(), lr=0.001)
total_steps_moe_grn = get_total_steps(train_loader, epochs)
warmup_steps_moe_grn = int(0.1 * total_steps_moe_grn)  # 10% of total steps
scheduler_moe_grn = get_cosine_schedule_with_warmup(optimizer_moe_grn, num_warmup_steps=warmup_steps_moe_grn, num_training_steps=total_steps_moe_grn)

train(convnext_moe_grn, train_loader, optimizer_moe_grn, scheduler_moe_grn, criterion, epochs=epochs)
test(convnext_moe_grn, test_loader)
