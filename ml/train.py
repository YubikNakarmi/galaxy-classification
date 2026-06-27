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
from src.training import CheckpointManager




root = Path(__file__).parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")


log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
