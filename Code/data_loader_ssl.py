# data_loader_ssl.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class UnlabeledDataset(Dataset):
    """دیتاست بدون برچسب برای Phase 1"""
    def __init__(self, root, img_size=224):
        self.root = root
        self.image_paths = []
        
        # فقط مسیر تصاویر - بدون برچسب!
        for class_name in os.listdir(root):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Loaded {len(self.image_paths)} UNLABELED images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

def get_ssl_dataloader(dataset_path, batch_size=32, shuffle=True):
    dataset = UnlabeledDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)