import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(batch_size=64, img_size=128, dataset_path='./data', train_ratio=0.8):
    # تشخیص خودکار مسیر کاگل
    if os.path.exists('/kaggle/input/cassava-leaf-disease-classification/data'):
        dataset_path = '/kaggle/input/cassava-leaf-disease-classification/data'
        print(f"✅ Kaggle dataset detected at: {dataset_path}")
    
    # تبدیلات آموزش
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
    
    # تبدیلات تست
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # بارگذاری کل دیتاست (بدون نیاز به پوشه train/test جدا)
    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    
    # تقسیم به train و test
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_data, test_data = random_split(full_dataset, [train_size, test_size])
    
    # اعمال transform جداگانه برای test
    test_data.dataset.transform = test_transform
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"📊 Found {len(train_data)} training images, {len(test_data)} test images")
    print(f"🏷️ Classes: {full_dataset.classes}")
    
    return train_loader, test_loader, full_dataset.classes