from __future__ import annotations

from src.utils.azure_utils import AzureBlobStorage
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.config import load_and_validate_config
from src.config.schema import RootConfig
from src.utils import setup_ml_logging
from src.utils import load_yaml
from loguru import logger

root = Path(__file__).parent.parent.parent
log_file = root / "configs" / "core.yaml"


@dataclass
class CheckpointState:
    start_epoch: int
    best_val_loss: float
    resume_run_id: Optional[str]

class CheckpointManager:

    '''Class for managing model checkpoints. It handles saving and loading checkpoints to/from local storage and Azure Blob Storage.'''


    def __init__(self, checkpoint_dir_local: str,checkpoint_dir_azure: 
                 str,cfg:RootConfig,run_id:str | None = None):

        self.checkpoint_dir = Path(checkpoint_dir_local) / "best.pt" #local dir
        self.checkpoint_dir_tmp = Path(checkpoint_dir_local) / "tmp.pt" #local dir for load

        self.checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir_azure = checkpoint_dir_azure
        self.cfg = cfg
        self.az_container = cfg.storage.cloud.experiments.container
        self.resume_hard = cfg.train.checkpoints.checkpoint_resume_hard

        self.resume_enabled = cfg.train.checkpoints.resume_from_checkpoint
        self.run_id = run_id 

    def save_checkpoint(self,checkpoint: dict):
        ''' Saves the model and optimizer state to a checkpoint file in local storage and uploads it to Azure Blob Storage. 

        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            scheduler: The scheduler to save.
            epoch: The current epoch number.

        '''

        torch.save(checkpoint, self.checkpoint_dir)

        logger.success(f"Checkpoint saved locally at {self.checkpoint_dir}")

    def upload_checkpoint_to_azure(self)->str| None:
        ''' Uploads the checkpoint file from local storage to Azure Blob Storage. '''

        if not self.run_id:
            raise RuntimeError("run_id must be set before uploading checkpoint.")

        try:
            az_ = AzureBlobStorage(container_name=self.az_container)
            path = az_.upload_blob(blob_name=self.checkpoint_dir_azure, data=self.checkpoint_dir)
            logger.success(f"Checkpoint uploaded to Azure Blob Storage at {self.checkpoint_dir_azure} in container {self.az_container}")

            return path
        except Exception as e:
            logger.exception(f"Error uploading checkpoint to Azure Blob Storage: {e}")  

            return None


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
        
        logger.warning(f"Attempting to resume from blob: {self.checkpoint_dir_azure} in container: {self.az_container}")
        try: 
            az_loader = AzureBlobStorage(container_name=self.az_container)
            az_loader.load_blob(blob_name= self.checkpoint_dir_azure, local_path=self.checkpoint_dir_tmp)

        except Exception as e:
            logger.exception(f"Error loading checkpoint from Azure Blob Storage: {e}")
            logger.warning("Starting training from scratch due to checkpoint load failure.")
            return self._handle_missing()
        
        ckpt = torch.load(self.checkpoint_dir_tmp, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
 
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        resume_run_id = ckpt.get("run_id")

        logger.success(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch} with best validation loss {best_val_loss}.")
        return CheckpointState(start_epoch=start_epoch, best_val_loss=best_val_loss, resume_run_id=resume_run_id)
        


        
    def _handle_missing(self) -> CheckpointState:
        if self.resume_hard:
            raise RuntimeError(
                "resume_from_checkpoint=True and checkpoint_resume_hard=True "
                "but no checkpoint found. Exiting."
            )
        logger.warning("No checkpoint found. Starting fresh training.")
        return CheckpointState(start_epoch=0, best_val_loss=float("inf"), resume_run_id=None)




