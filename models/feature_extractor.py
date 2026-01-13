"""
Feature Extractor with Spatial Attention
ResNet-34 backbone with spatial attention module as described in the paper.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from .attention import SpatialAttention
import sys
sys.path.append('..')
from config import FEATURE_DIM, ATTENTION_KERNEL, DROPOUT_RATE


class FeatureExtractor(nn.Module):
    """
    ResNet-34 backbone with spatial attention for feature extraction.
    
    From paper:
    - ResNet-34 backbone (convolutional layers up to conv5_x)
    - Spatial attention module
    - Global Average Pooling → 512-d feature vector
    """
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        # Load ResNet-34 backbone
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Extract convolutional layers (up to conv5_x)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # conv2_x
        self.layer2 = resnet.layer2  # conv3_x
        self.layer3 = resnet.layer3  # conv4_x
        self.layer4 = resnet.layer4  # conv5_x
        
        # Spatial attention module (7x7 conv + sigmoid)
        # Input channels = 512 (output of ResNet-34 layer4)
        self.attention = SpatialAttention(in_channels=512, kernel_size=ATTENTION_KERNEL)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Extract features from input image/patch.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            features: Feature vector of shape (B, 512)
            attention_map: Attention map of shape (B, 1, H', W')
        """
        # ResNet backbone forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, H', W')
        
        # Apply spatial attention
        x, attention_map = self.attention(x)  # (B, 512, H', W')
        
        # Global Average Pooling and flatten
        features = self.gap(x)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        return features, attention_map


class DefectClassifier(nn.Module):
    """
    Complete defect classification model.
    Feature Extractor + Classification Head
    
    Used for supervised pre-training stage.
    """
    def __init__(self, pretrained=True):
        super(DefectClassifier, self).__init__()
        
        # Feature extractor (ResNet-34 + Attention)
        self.feature_extractor = FeatureExtractor(pretrained=pretrained)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(256, 1)  # Binary classification (sigmoid applied in loss)
        )
        
    def forward(self, x):
        """
        Classify input image as defect/no-defect.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            logits: Classification logits of shape (B, 1)
            features: Feature vector of shape (B, 512)
            attention_map: Attention map
        """
        features, attention_map = self.feature_extractor(x)
        logits = self.classifier(features)
        
        return logits, features, attention_map
    
    def freeze_feature_extractor(self):
        """Freeze feature extractor weights for RL training"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def unfreeze_feature_extractor(self):
        """Unfreeze feature extractor weights"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
