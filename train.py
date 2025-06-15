import os
import time
import logging
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =============== Logging ===============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============== Hyperparametreler ===============
IMG_HEIGHT = 50
IMG_WIDTH = 200
NUM_DIGITS = 6
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4  # Daha düşük learning rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============== Dataset ===============
class CaptchaDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Basitleştirilmiş augmentation
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(3),  # Daha az rotasyon
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # Daha az kaydırma
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = str(self.df.iloc[idx]['solution'])[:6].zfill(NUM_DIGITS)
        
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            
            if self.is_train:
                image = self.aug_transform(image)
            else:
                image = self.transform(image)
            
            label_tensor = torch.tensor([int(c) for c in label], dtype=torch.long)
            return image, label_tensor
        except Exception as e:
            logger.error(f"Görüntü hatası: {img_path} → {e}")
            raise

# =============== Model ===============
class CaptchaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Daha basit ve etkili bir CNN mimarisi
        self.features = nn.Sequential(
            # İlk blok
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # İkinci blok
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Üçüncü blok
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Dördüncü blok
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Her rakam için ayrı sınıflandırıcı
        self.digit_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 3 * 12, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, NUM_CLASSES)
            ) for _ in range(NUM_DIGITS)
        ])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # Her rakam için ayrı tahmin
        outputs = []
        for classifier in self.digit_classifiers:
            outputs.append(classifier(x))
        
        return torch.stack(outputs, dim=1)

def train():
    df = pd.read_csv("captcha_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        CaptchaDataset(train_df, True), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        CaptchaDataset(val_df, False), 
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )
    
    model = CaptchaModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    best_val_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=2)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            
            # İlk 5 tahmini yazdır
            if epoch == 1 and total <= 5:
                for i in range(min(5, len(labels))):
                    target_str = ''.join(map(str, labels[i].cpu().numpy()))
                    pred_str = ''.join(map(str, preds[i].cpu().numpy()))
                    logger.info(f"Hedef: {target_str} | Tahmin: {pred_str}")
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=2)
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
        
        val_acc = 100. * val_correct / val_total
        logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Learning rate güncelleme
        scheduler.step(val_acc)
        
        # Model kaydetme
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, "captcha_model.pth")
            logger.info(f"Yeni en iyi model kaydedildi: {val_acc:.2f}%")

if __name__ == '__main__':
    train()
