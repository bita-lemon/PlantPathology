import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import get_dataloaders
from finetune_classifier import CNNClassifier
from utils import set_seed

def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # بارگذاری دیتا (بدون نیاز به مسیر جداگانه train/test)
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=32,
        img_size=224,
        dataset_path='/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data'
    )
    
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    model = CNNClassifier(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        total_loss, total_correct = 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/10")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=f"{100*total_correct/len(train_loader.dataset):.1f}%")

        acc = total_correct / len(train_loader.dataset)
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={100*acc:.2f}%")

    torch.save(model.state_dict(), "final_model.pth")
    print("✅ Model saved!")

if __name__ == "__main__":
    train_model()