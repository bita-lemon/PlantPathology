# simclr_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssl_encoder import SSLEncoder

class SimCLR(nn.Module):
    """مدل کامل SimCLR با Projection Head"""
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
    
    def get_encoder(self):
        return self.encoder