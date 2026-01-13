"""
DAGM Dataset Loader
Loads DAGM Class10 dataset with patch extraction support
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
from config import (DATA_DIR, IMAGE_SIZE, PATCH_SIZE, INPUT_SIZE, 
                    AUGMENTATION, BATCH_SIZE)


class DAGMDataset(Dataset):
    """
    DAGM 2007 Class10 Dataset for defect detection.
    
    Each sample consists of:
    - Grayscale image (512x512)
    - Binary label (0: no defect, 1: defect)
    - Ground-truth mask (if defect)
    """
    def __init__(self, root_dir, split='Train', transform=None, return_full_image=True):
        """
        Args:
            root_dir: Path to DAGM directory
            split: 'Train' or 'Test'
            transform: Optional transforms
            return_full_image: If True, return full image. If False, return patches.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_full_image = return_full_image
        
        # Path to Class10 directory
        self.class_dir = os.path.join(root_dir, 'Class10', split)
        
        # Load labels
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load sample paths and labels"""
        samples = []
        labels_file = os.path.join(self.class_dir, 'labels.txt')
        
        if os.path.exists(labels_file):
            # Load from labels file
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        label = int(parts[1])
                        mask_name = parts[2] if len(parts) > 2 else None
                        
                        img_path = os.path.join(self.class_dir, img_name)
                        mask_path = os.path.join(self.class_dir, 'Label', mask_name) if mask_name else None
                        
                        if os.path.exists(img_path):
                            samples.append({
                                'image': img_path,
                                'label': label,
                                'mask': mask_path
                            })
        else:
            # Auto-detect from directory structure
            img_dir = self.class_dir
            label_dir = os.path.join(self.class_dir, 'Label')
            
            for img_file in sorted(os.listdir(img_dir)):
                if img_file.endswith(('.png', '.PNG', '.jpg', '.bmp')):
                    img_path = os.path.join(img_dir, img_file)
                    
                    # Check for corresponding mask
                    base_name = os.path.splitext(img_file)[0]
                    mask_path = None
                    label = 0  # Default: no defect
                    
                    for ext in ['.png', '.PNG', '.bmp']:
                        potential_mask = os.path.join(label_dir, f'{base_name}_label{ext}')
                        if os.path.exists(potential_mask):
                            mask_path = potential_mask
                            label = 1
                            break
                    
                    samples.append({
                        'image': img_path,
                        'label': label,
                        'mask': mask_path
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (grayscale -> RGB for ResNet)
        image = Image.open(sample['image']).convert('RGB')
        label = sample['label']
        
        # Load mask if exists
        mask = None
        if sample['mask'] and os.path.exists(sample['mask']):
            mask = Image.open(sample['mask']).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'has_mask': mask is not None,
            'idx': idx
        }


class PatchExtractor:
    """
    Extract patches from images for RL agent exploration.
    
    From paper: At each time step t, the agent observes a fixed-size
    square patch extracted from the full image.
    """
    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Normalization transform for patches
        self.normalize = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_patch(self, image, u, v):
        """
        Extract a patch from the image at position (u, v).
        
        Args:
            image: PIL Image or tensor
            u: Top-left x coordinate
            v: Top-left y coordinate
            
        Returns:
            Normalized patch tensor
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL for cropping
            image = transforms.ToPILImage()(image)
        
        # Ensure coordinates are within bounds
        u = max(0, min(u, self.image_size - self.patch_size))
        v = max(0, min(v, self.image_size - self.patch_size))
        
        # Crop patch
        patch = image.crop((u, v, u + self.patch_size, v + self.patch_size))
        
        # Normalize
        patch_tensor = self.normalize(patch)
        
        return patch_tensor
    
    def get_center_position(self):
        """Get center position for initial patch"""
        center = (self.image_size - self.patch_size) // 2
        return center, center


def get_train_transform():
    """Get training transforms with augmentation"""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=AUGMENTATION['horizontal_flip']),
        transforms.RandomVerticalFlip(p=AUGMENTATION['vertical_flip']),
        transforms.RandomRotation(degrees=AUGMENTATION['rotation']),
        transforms.ColorJitter(
            brightness=AUGMENTATION['brightness'],
            contrast=AUGMENTATION['contrast']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transform():
    """Get test transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """Create train and test dataloaders"""
    train_dataset = DAGMDataset(
        root_dir=data_dir,
        split='Train',
        transform=get_train_transform()
    )
    
    test_dataset = DAGMDataset(
        root_dir=data_dir,
        split='Test',
        transform=get_test_transform()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader
