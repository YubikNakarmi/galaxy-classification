from __future__ import annotations

from src.utils.azure_utils import AzureBlobStorage
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.config import load_and_validate_config
from src.config.schema import RootConfig
from src.utils import setup_ml_logging
from loguru import logger

root = Path(__file__).parent.parent.parent
cfg_file = root / "configs" / "ml.yaml"
log_file = root / "configs" / "core.yaml"

setup_ml_logging(log_file)

cfg = load_and_validate_config(cfg_file,root = root)

@dataclass
class CheckpointState:
    start_epoch: int
    best_val_loss: float
    resume_run_id: Optional[str]

class CheckpointManager:

    '''Class for managing model checkpoints. It handles saving and loading checkpoints to/from local storage and Azure Blob Storage.'''


    def __init__(self, checkpoint_dir_local: str,checkpoint_dir_azure: 
                 str,config:RootConfig,resume_enabled:bool = False,run_id:str | None = None):

        self.checkpoint_dir = Path(checkpoint_dir_local) / "best.pt"
        self.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir_azure = checkpoint_dir_azure

        self.cfg = config
        self.az_container = cfg.storage.cloud.experiments.container
        self.resume_enabled = resume_enabled
        self.run_id = run_id 

    def save_checkpoint(self, model, optimizer, epoch):
        pass

    def load_checkpoint(self, model, optimizer,scheduler, device)-> CheckpointState:
        ''' Loads checkpoint to local storage from Azure Blob Storage if it exists, and then loads the model 
        and optimizer state from the checkpoint. 

        Args:
            model: The model to load the checkpoint into.
            optimizer: The optimizer to load the checkpoint into.
            scheduler: The scheduler to load the checkpoint into.
            device: The device to map the model and optimizer to.

        Returns:
            CheckpointState: A dataclass containing the start epoch, best validation loss, and resume run ID.

        '''

        if not self.resume_enabled:
            logger.info("Resume is disabled. Starting training from scratch.")
            return CheckpointState(start_epoch=0, best_val_loss=float('inf'), resume_run_id=None)
        
        try: 
            az_loader = AzureBlobStorage(container_name=self.az_container)
            az_loader.load_blob(blob_name= self.checkpoint_dir_azure, local_path=self.checkpoint_dir)



