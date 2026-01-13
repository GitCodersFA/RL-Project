"""
Spatial Attention Module
As described in the paper: 7x7 convolution with sigmoid activation
"""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for feature map modulation.
    
    Uses a 7x7 convolution followed by sigmoid activation to produce
    a 2D attention map that highlights salient regions.
    
    From paper:
    A_t = σ(W_a * F_t + b_a) ∈ R^(1×H'×W')
    where * is 7×7 convolution, σ the sigmoid
    """
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # 7x7 convolution to produce attention map
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: Feature map tensor of shape (B, C, H, W)
            
        Returns:
            Modulated feature map of shape (B, C, H, W)
            Attention map of shape (B, 1, H, W)
        """
        # Compute attention map
        attention = self.sigmoid(self.conv(x))  # (B, 1, H, W)
        
        # Apply attention to modulate features (element-wise multiplication)
        modulated = x * attention  # (B, C, H, W)
        
        return modulated, attention
