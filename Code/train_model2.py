# train_model.py - نسخه جدید با اتصال به Phase 1
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import get_dataloaders
from finetune_classifier import CNNClassifier
from utils import set_seed
import os

def train_model(use_phase1_encoder=True, encoder_path='ssl_encoder.pth'):
    """
    آموزش مدل با قابلیت استفاده از encoder Phase 1
    
    Args:
        use_phase1_encoder: اگر True باشد، از encoder Phase 1 استفاده می‌کند
        encoder_path: مسیر فایل encoder آموزش دیده در Phase 1
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\n" + "="*50)
    if use_phase1_encoder and os.path.exists(encoder_path):
        print("🌱 PLANTCLR: Phase 2 with Phase 1 Pretrained Encoder")
        print(f"   Encoder loaded from: {encoder_path}")
    else:
        print("🌱 Training from scratch (no Phase 1 encoder)")
    print("="*50)

    # بارگذاری دیتا
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=32,
        img_size=224,
        dataset_path='/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data'
    )
    
    print(f"\nClasses: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # ساخت مدل
    if use_phase1_encoder and os.path.exists(encoder_path):
        model = CNNClassifier(
            num_classes=len(class_names),
            use_pretrained_encoder=True,
            encoder_path=encoder_path
        ).to(device)
    else:
        model = CNNClassifier(num_classes=len(class_names)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    
    for epoch in range(1, 31):
        model.train()
        total_loss, total_correct = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/30")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()
            
            loop.set_postfix(
                loss=loss.item(), 
                acc=f"{100*total_correct/len(train_loader.dataset):.1f}%"
            )

        train_acc = total_correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Train Acc={100*train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_correct += (output.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_loader.dataset)
        print(f"   Validation Acc: {100*val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "final_model.pth")
            print(f"   💾 Best model saved! (Acc: {best_acc*100:.2f}%)")

    print(f"\n✅ Model saved! Best validation accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    # اگر فایل encoder Phase 1 وجود دارد، از آن استفاده کن
    encoder_path = 'ssl_encoder.pth'
    
    if os.path.exists(encoder_path):
        print("📁 Phase 1 encoder found! Using it for Phase 2...")
        train_model(use_phase1_encoder=True, encoder_path=encoder_path)
    else:
        print("⚠️ No Phase 1 encoder found. Training from scratch...")
        print("   (To use Phase 1, first run phase1_pretrain.py)")
        train_model(use_phase1_encoder=False)