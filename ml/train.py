import click
import torch
import pandas as pd
import src.dataset.dataset as dataset
import src.models.model as model
from src.dataset.dataset import GalaxyDataset
from torch.utils.data import DataLoader, dataset
from torchvision import transforms
import mlflow
import random
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
from src.models.model import GalaxyClassifier_ResNet
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from loguru import logger

root = Path(__file__).parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]

device = torch.device(dev)

epoch = int(ml_yaml["train"]["num_epochs"])
image_dir = ml_yaml["data"]["img_dir"]
split_dir = ml_yaml["data"]["split_dir"]
batch_size = int(ml_yaml["train"]["batch_size"])

train_path = split_dir / "train_split.csv"
val_path = split_dir / "val_split.csv"


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



def train_default():

    train_label_df = pd.read_csv(train_path)
    val_label_df = pd.read_csv(val_path)

    train_dataset = GalaxyDataset(labels_df=train_label_df, img_dir=image_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=val_label_df, img_dir=image_dir, transform=raw_transforms)

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()
    val_sampler, val_class_weights = val_dataset.make_weighted_sampler()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,sampler=train_sampler)
    
    model = GalaxyClassifier_ResNet(num_classes=5)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)


    global_step = 0
    for epoch in range(10):
        model.train()

        pbar = tqdm(subset_dataloader, desc=f"Epoch {epoch+1}")

        for images, labels in subset_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            optimizer.step()
            scheduler.step(epoch+ global_step/len(subset_dataloader))
            
            global_step += 1

            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })


        



def main():





