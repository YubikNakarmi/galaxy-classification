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

core_cfg = load_yaml(root / 'configs' / 'core.yaml')

file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config)

#used throughout all loggings runs for local saves, cloud saves use mlflow run_id
experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  
dev = cfg.env.device
active_variant = cfg.active_variant

validate_setup(required_paths=[cfg.data.img_dir,cfg.data.train_path,cfg.data.val_path], 
               mlflow_tracking_uri=MLFLOW_TRACKING_URI,)

def make_metrics(num_classes:int, device: torch.device) -> dict:
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

def register_model(model,val_loader:DataLoader):
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images.to(dev)
    
    # getting model signatures and sample for pt2 serialization
    with torch.no_grad():
        preds_np = torch.softmax(model(sample_images), dim=1).cpu().numpy()
 
    signature = infer_signature(sample_images.cpu().numpy(), preds_np)
 
    mlflow.pytorch.log_model(
        model,
        serialization_format="pt2", #most robust format for torchscript and future compatibility
        input_example=sample_images.cpu(),
        name="model",
        registered_model_name=cfg.variant.mlflow.model_name,
        signature=signature,
    )
    logger.info("Model logged to MLflow registry.")
 
def upload_artifact(blob: AzureBlobStorage, run_id: str)-> None:
    log_path = root / core_cfg["logging"]["ml"]["file"]
 
    with open(log_path, "rb") as f:
        url = blob.upload_blob(blob_name=f"{run_id}/logs/training.log", data=f)
    mlflow.set_tag("log_url", url)
 
    try:
        report_path = cfg.storage.local.logs.reports_dir / f"{experiment_name}.md"
        with open(report_path, "rb") as f:
            url = blob.upload_blob(blob_name=f"{run_id}/reports/report.md", data=f)
        mlflow.set_tag("report_url", url)
    except Exception as e:
        logger.error(f"Could not upload report: {e}")
 
    try:
        tb_run_dir = cfg.storage.local.logs.tensorboard_dir / experiment_name
        for file in tb_run_dir.iterdir():
            if file.is_file():
                with open(file, "rb") as f:
                    url = blob.upload_blob(blob_name=f"{run_id}/tensorboard/{file.name}", data=f)
                mlflow.set_tag(f"tensorboard_{file.name}", url)
    except Exception as e:
        logger.error(f"Could not upload tensorboard logs: {e}")


def train_default(variant:str, resume_enabled:bool, hard_resume:bool,):

    logger.info(f"Starting training for variant: {variant} with resume_enabled={resume_enabled} and hard_resume={hard_resume}")
    random.seed(cfg.env.seed)
    torch.manual_seed(cfg.env.seed)
    torch.cuda.manual_seed_all(cfg.env.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    train_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.train_path), img_dir=cfg.data.img_dir, transform=raw_transforms)
    val_dataset = GalaxyDataset(labels_df=pd.read_csv(cfg.data.val_path), img_dir=cfg.data.img_dir, transform=val_transforms)

    train_sampler, train_class_weights = train_dataset.make_weighted_sampler()

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=False,sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    #load models and optimizers
    assert cfg.variant.num_classes is not None, \
    f"variant '{cfg.active_variant}' has num_classes=null, cannot use classification model"

    model = GalaxyClassifier_ResNet(num_classes=cfg.variant.num_classes)
    model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate) #weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.scheduler.warmup_epochs,eta_min = cfg.train.eta_min ,T_mult=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_class_weights).to(dev))
    
    load_chkpt = CheckpointManager(config=cfg,checkpoint_dir_local=cfg.train.checkpoints.local_checkpoint_path,
                                   checkpoint_dir_azure=cfg.storage.cloud.checkpoints.blob_path,resume_enabled=resume_enabled,run_id=None)

    pass
