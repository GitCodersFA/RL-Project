"""
Utility functions for training and evaluation
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, accuracy, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('loss'), checkpoint.get('accuracy')


def plot_training_curves(train_losses, train_accs, val_accs, save_path=None):
    """Plot training curves as shown in Figure 2 of the paper"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, classes=['No Defect', 'Defect'], save_path=None):
    """Plot confusion matrix as shown in Figure 3 of the paper"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved: {save_path}")
    
    plt.show()


def plot_rl_rewards(episode_rewards, save_path=None):
    """Plot RL training rewards"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    episodes = range(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
    
    # Moving average
    window = min(50, len(episode_rewards) // 4)
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episode_rewards) + 1), moving_avg, 'r-', 
                linewidth=2, label=f'Moving Avg ({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('RL Training - Episode Rewards')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"RL rewards plot saved: {save_path}")
    
    plt.show()


def get_timestamp():
    """Get timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class AverageMeter:
    """Compute and store running average"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
