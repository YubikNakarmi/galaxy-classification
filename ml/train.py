import importlib
import torch
import os
import pandas as pd
import src.dataset.dataset as dataset
import src.models.model as model
from src.dataset.dataset import GalaxyDataset
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
import yaml
import mlflow
import random

raw_transforms = transforms.Compose([
    transforms.CenterCrop(random.randint(170, 240)),  
    transforms.Resize(160),                            
    transforms.RandomCrop(128),                        
    transforms.RandomRotation(degrees = [90, 180]),
    transforms.RandomRotation(degrees = [0,270]),      
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])


def train():
    for epoch in range(10):
        model.train()

        pbar = tqdm(subset_dataloader, desc=f"Epoch {epoch+1}")

        for images, labels in subset_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })
        
        scheduler.step()
def main():



