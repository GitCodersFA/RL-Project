# RL-Driven Garment Defect Detection

Implementation of **"Reinforcement Learning-Driven Real-Time Garment Defect Detection with Adaptive Visual Attention Mechanisms"** (QPAIN 2025).

This framework combines:
- **ResNet-34 CNN** with spatial attention for feature extraction
- **Actor-Critic (Policy Gradient)** agent for intelligent patch-based inspection
- **Two-stage training**: Supervised pre-training → RL fine-tuning

---

## 📁 Project Structure

```
├── config.py                 # Hyperparameters and configuration
├── dataset.py                # DAGM dataset loader with patch extraction
├── download_dataset.py       # Download DAGM 2007 dataset
├── train_supervised.py       # Stage 1: Supervised pre-training
├── train_rl.py               # Stage 2: RL training
├── evaluate.py               # Evaluation and visualization
├── utils.py                  # Utility functions
├── models/
│   ├── attention.py          # Spatial attention module (7x7 conv)
│   └── feature_extractor.py  # ResNet-34 backbone + classifier
├── rl/
│   ├── environment.py        # MDP environment for defect detection
│   ├── actor_critic_agent.py # Actor-Critic (A2C-style) agent (on-policy)
│   └── replay_buffer.py      # Experience replay buffer
├── data/                     # Dataset directory
└── checkpoints/              # Saved models
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_dataset.py
```

Downloads the **DAGM 2007 Class10** dataset (~5.4GB).

### 3. Run Supervised Pre-training (Stage 1)

```bash
# Quick test (2 epochs)
python train_supervised.py --epochs 2

# Full training (20 epochs as in paper)
python train_supervised.py --epochs 20
```

### 4. Run RL Training (Stage 2)

```bash
# Quick test (50 episodes)
python train_rl.py --episodes 50

# Full training (400 episodes as in paper)
python train_rl.py --episodes 400
```

### 5. Evaluate

```bash
python evaluate.py
```

---

## ⚙️ Command-Line Arguments

### Supervised Training (`train_supervised.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--lr` | 0.0001 | Learning rate |
| `--no_plot` | False | Disable matplotlib plots |

**Examples:**
```bash
# Train with 5 epochs
python train_supervised.py --epochs 5

# Custom batch size and learning rate
python train_supervised.py --epochs 10 --batch_size 16 --lr 0.0001

# Headless server (no display)
python train_supervised.py --epochs 20 --no_plot
```

### RL Training (`train_rl.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 400 | Number of training episodes |
| `--target_update` | 10 | (Not used by Actor-Critic; legacy flag kept for compatibility) |
| `--no_plot` | False | Disable matplotlib plots |

**Examples:**
```bash
# Train with 100 episodes
python train_rl.py --episodes 100

# Custom target network update frequency
python train_rl.py --episodes 200 --target_update 5

# Headless server (no display)
python train_rl.py --episodes 400 --no_plot
```

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~100% |
| Precision (Defect) | 1.00 |
| Recall (Defect) | 1.00 |
| Inference Time (GPU) | ~3.7 ms |
| Inference Time (CPU) | ~28 ms |

---

## 🧪 Testing Individual Components

### Test Dataset Loader
```python
from dataset import get_dataloaders
train_loader, test_loader = get_dataloaders()
print(f"Train samples: {len(train_loader.dataset)}")
```

### Test Feature Extractor
```python
import torch
from models.feature_extractor import DefectClassifier

model = DefectClassifier(pretrained=True)
x = torch.randn(1, 3, 224, 224)
logits, features, attention = model(x)
print(f"Features shape: {features.shape}")  # (1, 512)
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--batch_size` |
| Slow training | Use GPU or reduce `--epochs` |
| Dataset not found | Run `python download_dataset.py` |
| Matplotlib error | Add `--no_plot` flag |

---

## 📝 Paper Reference

```bibtex
@inproceedings{sobhan2025rl,
  title={Reinforcement Learning-Driven Real-Time Garment Defect Detection 
         with Adaptive Visual Attention Mechanisms},
  author={Sobhan, Abdus and Sourav, Md Tanvir Islam and 
          Bassam, Abdullah Al and Masud, Nasrullah},
  booktitle={QPAIN},
  year={2025}
}
```
