"""
Supervised Pre-training Script
Train the feature extractor + classifier
As described in Section IV.5 of the paper.

Usage:
    python train_supervised.py --epochs 20
    python train_supervised.py --epochs 5 --batch_size 16
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    DEVICE, DATA_DIR, SUPERVISED_EPOCHS, SUPERVISED_LR, 
    WEIGHT_DECAY, BATCH_SIZE
)
from dataset import get_dataloaders
from models.feature_extractor import DefectClassifier
from utils import save_checkpoint, plot_training_curves, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised Pre-training for Defect Detection')
    parser.add_argument('--epochs', type=int, default=SUPERVISED_EPOCHS,
                        help=f'Number of training epochs (default: {SUPERVISED_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=SUPERVISED_LR,
                        help=f'Learning rate (default: {SUPERVISED_LR})')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting (useful for headless servers)')
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        
        # Forward pass
        logits, _, _ = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        loss_meter.update(loss.item(), images.size(0))
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    return loss_meter.avg, 100 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits, _, _ = model(images)
            loss = criterion(logits, labels.unsqueeze(1))
            total_loss += loss.item() * images.size(0)
            
            predictions = (torch.sigmoid(logits) > 0.5).float().squeeze()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    args = parse_args()
    
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    
    print("=" * 60)
    print("Supervised Pre-training Stage")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Batch Size: {batch_size}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(DATA_DIR, 'Class10')):
        print("\nDataset not found! Running download script...")
        from download_dataset import download_dagm_dataset
        download_dagm_dataset()
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, test_loader = get_dataloaders(DATA_DIR, batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = DefectClassifier(pretrained=True).to(DEVICE)
    
    # Loss function (Binary Cross-Entropy)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Evaluate
        val_results = evaluate(model, test_loader, criterion, DEVICE)
        val_acc = val_results['accuracy']
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%, Precision: {val_results['precision']:.4f}, "
              f"Recall: {val_results['recall']:.4f}, F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_acc,
                'checkpoints/best_classifier.pth'
            )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load('checkpoints/best_classifier.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"\nTest Accuracy: {final_results['accuracy']:.2f}%")
    print(f"Precision (Defect Class): {final_results['precision']:.4f}")
    print(f"Recall (Defect Class): {final_results['recall']:.4f}")
    print(f"F1-Score (Defect Class): {final_results['f1']:.4f}")
    
    # Plot and save training curves
    os.makedirs('checkpoints', exist_ok=True)
    if not args.no_plot:
        try:
            plot_training_curves(train_losses, train_accs, val_accs, 'checkpoints/training_curves.png')
        except Exception as e:
            print(f"Could not generate plot: {e}")
    
    print("\n" + "=" * 60)
    print("Supervised pre-training complete!")
    print(f"Best model saved to: checkpoints/best_classifier.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()
