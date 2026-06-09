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
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
from tqdm import tqdm

import datetime


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

train_path = Path(split_dir) / "train_split.csv"
val_path = Path(split_dir) / "val_split.csv"

experiment_root = ml_yaml["variants"][ml_yaml["active_variant"]]["experiment"]["root"]
experiment_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")


writer = SummaryWriter(log_dir=root / "logs" / "tensorboard")

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

def validation()-> bool:
    
    if not Path.exists(train_path) and not Path.exists(val_path) and \
    not Path.exists(image_dir) and not Path.exists(split_dir):
        logger.error(f"Train split file {train_path} does not exist.")
        return False
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Connection Test")
        with mlflow.start_run(run_name="Validation Run"):
            mlflow.log_param("validation", "success")
            mlflow.log_metric("validation_metric", 1.0)

    except Exception as e:
        logger.error(f"MLflow tracking URI {MLFLOW_TRACKING_URI} is not accessible: {e}")
        return False

    return True


def train_default():
    
    #load data and samplers
    train_label_df = pd.read_csv(train_path)
    val_label_df = pd.read_csv(val_path)

    train_dataset = GalaxyDataset(labels_df=train_label_df, img_dir=image_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=val_label_df, img_dir=image_dir, transform=raw_transforms)

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()
    val_sampler = val_dataset.make_weighted_sampler()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,sampler=val_sampler)
    
    #load models and optimizers
    model = GalaxyClassifier_ResNet(num_classes=5)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(device))

    global_step = 0 #for scheeduler 

    for epoch in range(10):
        model.train()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for images, labels in train_dataloader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)# Gradient clipping to prevent exploding gradients


            optimizer.step()
            scheduler.step(epoch+ global_step/len(train_dataloader)) # Update scheduler with epoch and batch progress
            
            global_step += 1

            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_step)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"]
            })

        model.eval()
        val_loss = 0.0
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_loss /= len(val_dataloader.dataset)
                accuracy

        
@click.command()
@click.option("--variant", type=str, default="default", help="Training variant to use")


def main(variant):
    validation()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if variant == "default":
        mlflow.set_experiment(experiment_name)
        train_default()
    



if __name__ == "__main__":
    main()
