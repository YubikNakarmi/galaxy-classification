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


    def __init__(self, checkpoint_dir_local: str|Path,checkpoint_dir_azure: 
                 str,cfg:RootConfig,run_id:str | None = None):

        self.checkpoint_dir = Path(checkpoint_dir_local) #local

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir_azure = checkpoint_dir_azure
        self.cfg = cfg
        self.az_container = cfg.storage.cloud.experiments.container 
        self.resume_hard = cfg.train.checkpoints.checkpoint_resume_hard

        self.resume_enabled = cfg.train.checkpoints.resume_from_checkpoint
        self.run_id = run_id 

    def save(self, model, optimizer, scheduler, epoch: int, val_loss: float,
             val_f1_macro: float, cfg_snapshot: dict,filename:str|Path) -> None:
        
        """Save checkpoint locally. Upload to Azure if run_id is set.
            Args:
                model: The model to save.
                optimizer: The optimizer to save.
                scheduler: The scheduler to save.
                epoch: The current epoch number.
                val_loss: The validation loss.
                val_f1_macro: The validation F1 score (macro).
                cfg_snapshot: A dictionary containing the configuration snapshot.

        """
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_f1_macro": val_f1_macro,
            "run_id": self.run_id,
            "config": cfg_snapshot,
        }, self.checkpoint_dir / Path(filename))

        logger.success(f"Checkpoint saved locally (val_loss={val_loss:.4f})")

    def upload_checkpoint_to_azure(self,filename:str| Path, blob_path:Optional[str|Path]=None)->str| None:
        ''' Uploads the checkpoint file from local storage to Azure Blob Storage. '''

        if not self.run_id:
            raise RuntimeError("run_id must be set before uploading checkpoint.")

        try:

            with AzureBlobStorage(container_name=self.az_container) as az_:
                # upload local checkpoint file to blob storage under a run-specific path
                if blob_path is None:
                    blob_name = str(Path(self.checkpoint_dir_azure) / Path(self.run_id) / Path(filename))
                else:
                    blob_name = str(Path(blob_path))

                path = az_.upload_blob(blob_name=blob_name, data=self.checkpoint_dir / Path(filename), Overwrite=True)
                logger.success(f"Checkpoint uploaded to Azure Blob Storage at {blob_name} in container {self.az_container}")
                return path
           

        except Exception as e:
            logger.exception(f"Error uploading checkpoint to Azure Blob Storage: {e}")  

            return None


    def load_checkpoint(self, model, optimizer,scheduler, device,filename:str|Path="temp.pt", blob_name:Optional[str|Path]=None)-> CheckpointState:
        ''' Loads checkpoint to local storage from Azure Blob Storage if it exists, and then loads the model 
        and optimizer state from the checkpoint. 

        Args:
            model: The model to load the checkpoint into.
            optimizer: The optimizer to load the checkpoint into.
            scheduler: The scheduler to load the checkpoint into.
            device: The device to map the model and optimizer to.
            filename: The name of the checkpoint file to load.
            blob_name: The name of the blob in Azure Blob Storage to load the checkpoint from.

        Returns:
            CheckpointState: A dataclass containing the start epoch, best validation loss, and resume run ID.

        '''

        if not self.resume_enabled:
            logger.info("Resume is disabled. Starting training from scratch.")
            return CheckpointState(start_epoch=0, best_val_loss=float('inf'), resume_run_id=None)
        
        logger.warning(f"Attempting to resume from blob: {self.checkpoint_dir_azure} in container: {self.az_container}")
        try:
            if not self.run_id:
                raise RuntimeError("run_id must be set before loading checkpoint.")
            
            with AzureBlobStorage(container_name=self.az_container) as az_loader:
                if blob_name is None:
                    blob_name = str(Path(self.run_id) / self.checkpoint_dir_azure / Path(filename))
                else:
                    blob_name = str(Path(blob_name))
                logger.info(f"Loading checkpoint from Azure Blob Storage: {blob_name}")
                az_loader.load_blob(blob_name=blob_name, local_path=self.checkpoint_dir / Path(filename))
                logger.success(f"Checkpoint downloaded from Azure Blob Storage to {self.checkpoint_dir / Path(filename)}")

        except Exception as e:
            logger.exception(f"Error loading checkpoint from Azure Blob Storage: {e}")
            logger.warning("Starting training from scratch due to checkpoint load failure.")
            return self._handle_missing()
        
        ckpt = torch.load(self.checkpoint_dir / Path(filename), map_location=device)
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




