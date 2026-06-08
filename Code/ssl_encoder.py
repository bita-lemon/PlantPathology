# ssl_encoder.py
import torch
import torch.nn as nn

class SSLEncoder(nn.Module):
    """Encoder برای استخراج ویژگی - خروجی این مدل در Phase 2 استفاده می‌شود"""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, feature_dim, 3, padding=1), nn.BatchNorm2d(feature_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        return features.view(features.size(0), -1)