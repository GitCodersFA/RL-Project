"""
Evaluation Script
Evaluate the trained model and generate visualizations.
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import time

from config import DEVICE, DATA_DIR, BATCH_SIZE
from dataset import get_dataloaders
from models.feature_extractor import DefectClassifier
from utils import plot_confusion_matrix


def evaluate_classifier(model, test_loader, device):
    """Evaluate classifier and generate metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            
            # Time inference
            start = time.time()
            logits, features, attention = model(images)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            inference_times.append((end - start) / images.size(0) * 1000)  # ms per image
            
            predictions = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
            
            # Handle single sample case
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            
            all_preds.extend(predictions)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_inference_time = np.mean(inference_times)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'avg_inference_time_ms': avg_inference_time
    }


def visualize_attention(model, dataloader, device, save_dir='checkpoints', num_samples=4):
    """Visualize attention maps on sample images"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    sample_count = 0
    
    for batch in dataloader:
        if sample_count >= num_samples:
            break
            
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        
        with torch.no_grad():
            logits, features, attention = model(images)
        
        for i in range(min(images.size(0), num_samples - sample_count)):
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            
            # Attention map
            attn = attention[i].squeeze().cpu().numpy()
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            
            # Prediction
            pred = (torch.sigmoid(logits[i]) > 0.5).item()
            label_text = "Defect" if labels[i] == 1 else "No Defect"
            pred_text = "Defect" if pred == 1 else "No Defect"
            
            # Plot
            row = sample_count
            
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f'Input (GT: {label_text})')
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(attn, cmap='hot')
            axes[row, 1].set_title('Attention Map')
            axes[row, 1].axis('off')
            
            # Overlay
            attn_resized = np.array(Image.fromarray(
                (attn * 255).astype(np.uint8)
            ).resize((224, 224))) / 255.0
            overlay = img.copy()
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.3 * attn_resized, 0, 1)
            
            axes[row, 2].imshow(overlay)
            axes[row, 2].set_title(f'Pred: {pred_text}')
            axes[row, 2].axis('off')
            
            sample_count += 1
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'attention_visualization.png')
    plt.savefig(save_path, dpi=150)
    print(f"Attention visualization saved: {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load model
    classifier_path = 'checkpoints/best_classifier.pth'
    if not os.path.exists(classifier_path):
        print("Trained model not found!")
        print("Please run train_supervised.py first.")
        return
    
    print("\nLoading model...")
    model = DefectClassifier(pretrained=False).to(DEVICE)
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("\nLoading test data...")
    _, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating classifier...")
    results = evaluate_classifier(model, test_loader, DEVICE)
    
    # Print results (Table 1 from paper)
    print("\n" + "=" * 60)
    print("Classification Performance (Table 1)")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 40)
    print(f"{'Test Accuracy':<30} {results['accuracy']:.2f}%")
    print(f"{'Precision (Defect Class)':<30} {results['precision']:.4f}")
    print(f"{'Recall (Defect Class)':<30} {results['recall']:.4f}")
    print(f"{'F1-Score (Defect Class)':<30} {results['f1']:.4f}")
    print(f"{'Avg Inference Time':<30} {results['avg_inference_time_ms']:.2f} ms")
    
    # Plot confusion matrix (Figure 3)
    print("\nGenerating confusion matrix (Figure 3)...")
    os.makedirs('checkpoints', exist_ok=True)
    plot_confusion_matrix(results['confusion_matrix'], save_path='checkpoints/confusion_matrix.png')
    
    # Visualize attention maps
    print("\nGenerating attention visualizations...")
    visualize_attention(model, test_loader, DEVICE, num_samples=4)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
