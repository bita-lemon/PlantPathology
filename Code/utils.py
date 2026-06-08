import torch
import random
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_accuracy(preds, labels):
    return (preds == labels).mean()


def load_pretrained_encoder(encoder_path, feature_dim=512):
    from ssl_encoder import SSLEncoder
    encoder = SSLEncoder(feature_dim=feature_dim)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    return encoder