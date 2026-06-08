# ssl_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        similarity = torch.matmul(z, z.T) / self.temperature
        
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity = similarity.masked_fill(mask, -1e9)
        
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0).to(z.device)
        
        return F.cross_entropy(similarity, labels)