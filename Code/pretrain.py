# pretrain.py
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config_ssl import config
from data_loader_ssl import get_ssl_dataloader
from ssl_encoder import SSLEncoder
from SimCLR_CNNClassifier import SimCLR
from ssl_loss import NTXentLoss
from utils import set_seed

def phase1_pretrain():
    print("\n" + "="*60)
    print("🔬 PHASE 1: Self-Supervised Pretraining")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    
    # دیتالودر
    train_loader = get_ssl_dataloader(config.dataset_path, config.batch_size)
    
    # مدل
    encoder = SSLEncoder(feature_dim=config.feature_dim)
    model = SimCLR(encoder, projection_dim=config.projection_dim).to(device)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                 momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = NTXentLoss(temperature=config.temperature)
    
    losses = []
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()
        
        print(f"📊 Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # ذخیره encoder
    torch.save(encoder.state_dict(), config.save_path)
    print(f"\n✅ Encoder saved to: {config.save_path}")
    
    # رسم نمودار
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Phase 1: Self-Supervised Learning Loss')
    plt.savefig('phase1_loss.png')
    plt.show()
    
    return encoder

if __name__ == "__main__":
    phase1_pretrain()