import argparse
import logging
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from model.convnextv2 import ConvNeXtV2
from model.convnextv2_moe import ConvNeXtV2_MoE
from model.convnextv2_moe_grn import ConvNeXtV2_MoE_GRN

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate models on CIFAR-10.')
    parser.add_argument('--model', type=str, default='convnext', choices=['convnext', 'convnext_moe', 'convnext_moe_grn'],
                        help='Name of the model to train and evaluate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--lambda_cov', type=float, default=0.1, help='Coefficient for auxiliary loss (if any).')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    args = parser.parse_args()
    return args

# Training function
def train(model, train_loader, optimizer, scheduler, criterion, device, args):
    model.to(device)
    model.train()
    scaler = GradScaler()
    total_steps = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader, 1), total=total_steps, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                if isinstance(model, (ConvNeXtV2_MoE)):
                    outputs, l_aux = model(images)
                    loss = criterion(outputs, labels) + args.lambda_cov * l_aux
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{current_lr:.6f}"
            })

        avg_loss = total_loss / total_steps
        logger.info(f"Epoch [{epoch}/{args.epochs}] completed. Average Loss: {avg_loss:.4f}")

# Test function with detailed metrics
def test(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            if isinstance(model, (ConvNeXtV2_MoE)):
                outputs, _ = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Test Precision: {precision * 100:.2f}%")
    logger.info(f"Test Recall: {recall * 100:.2f}%")
    logger.info(f"Test F1 Score: {f1_score * 100:.2f}%")

    return accuracy, precision, recall, f1_score

# Main function
def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model based on args.model
    if args.model == 'convnext':
        model = ConvNeXtV2(num_classes=10)
    elif args.model == 'convnext_moe':
        model = ConvNeXtV2_MoE(num_classes=10)
    elif args.model == 'convnext_moe_grn':
        model = ConvNeXtV2_MoE_GRN(num_classes=10)
    else:
        logger.error(f"Model '{args.model}' is not recognized.")
        return

    logger.info(f"Initialized model: {args.model}")

    # Print model summary (optional)
    from torchinfo import summary
    summary(model, input_size=(args.batch_size, 3, 32, 32), depth=3, col_names=["input_size", "output_size", "num_params"])

    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Start training
    logger.info("Starting training...")
    train(model, train_loader, optimizer, scheduler, criterion, device, args)

    # Start testing
    logger.info("Starting testing...")
    test(model, test_loader, device)

if __name__ == '__main__':
    main()
