import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(batch_size=64, img_size=128, dataset_path='./data'):
    # تشخیص خودکار مسیر کاگل
    if os.path.exists('/kaggle/input/cassava-leaf-disease-classification/data'):
        dataset_path = '/kaggle/input/cassava-leaf-disease-classification/data'
        print(f"✅ Kaggle dataset detected at: {dataset_path}")
    
    # تبدیلات آموزش (با Augmentation قوی‌تر)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # تبدیلات تست (فقط resize و normalize)
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # بارگذاری داده‌ها
    train_data = datasets.ImageFolder(f'{dataset_path}/train', transform=train_transform)
    test_data = datasets.ImageFolder(f'{dataset_path}/test', transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"📊 Found {len(train_data)} training images, {len(test_data)} test images")
    print(f"🏷️ Classes: {train_data.classes}")
    
    return train_loader, test_loader, train_data.classes