from __future__ import annotations
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger
 

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    writer: SummaryWriter,
    epoch: int,
    num_epochs: int,
    global_step: int,
    grad_clip: float,
    device: torch.device) -> tuple[float, int]:
    
    """
        Run one training epoch.
    
        Returns:
            (train_loss, updated_global_step)
    """
 
    pass

def validate_epoch():
    pass