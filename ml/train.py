import click
import torch
import pandas as pd
from torchvision import transforms
import mlflow
import random
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
import os
from loguru import logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from tqdm import tqdm
import datetime
from mlflow.models.signature import infer_signature

from src.dataset.utils import generate_report
from src.dataset.utils import RandomCenterCrop
from src.utils.azure_utils import AzureBlobStorage
from src.models.model import GalaxyClassifier_ResNet
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from src.dataset.dataset import GalaxyDataset

from src.config import load_and_validate_config
from src.training import CheckpointManager,validate_setup


root = Path(__file__).parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")

log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)

file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config)

#used throughout all loggings runs for local saves, cloud saves use mlflow run_id
experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  
dev = cfg.env.device

validate_setup(required_paths=[cfg.data.img_dir,cfg.data.train_path,cfg.data.val_path,], 

               mlflow_tracking_uri=MLFLOW_TRACKING_URI,)


def make_metrics(num_classes, device):
    return {
        "accuracy":           Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "f1_micro":           F1Score(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "f1_macro":           F1Score(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "f1_classwise":       F1Score(task="multiclass", average="none", num_classes=num_classes).to(device),
        "precision_micro":    Precision(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "precision_macro":    Precision(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "precision_classwise":Precision(task="multiclass", average="none", num_classes=num_classes).to(device),
        "recall_micro":       Recall(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "recall_macro":       Recall(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "recall_classwise":   Recall(task="multiclass", average="none", num_classes=num_classes).to(device),
        "roc_auc":            AUROC(task="multiclass", num_classes=num_classes).to(device),
    }

# Hardcoded as there will be less editing of these transforms, but can be moved to config if needed
raw_transforms = transforms.Compose([
    RandomCenterCrop(170, 240), 
    transforms.Resize(160),                            
    transforms.RandomCrop(128),                        
    transforms.RandomRotation(degrees = [0, 360]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(207),  
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])


def reset_metrics(metrics: dict): #reset all metrics at the end of each epoch
    for m in metrics.values():
        m.reset()
