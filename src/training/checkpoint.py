''' Checkpoint management for model training.
This module provides the CheckpointManager class, which handles saving and loading model checkpoints to/from local'''


from __future__ import annotations
from src.utils.azure_utils import AzureBlobStorage
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.config.schema import RootConfig
from src.utils import setup_ml_logging
from src.utils import load_yaml
from loguru import logger

root = Path(__file__).parent.parent.parent
log_file = root / "configs" / "core.yaml"
setup_ml_logging(load_yaml(log_file))  # Load logging configuration from core.yaml
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
             val_f1_macro: float, cfg_snapshot: dict,filename:str|Path="best.pt") -> None:
        
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
        target = self.checkpoint_dir / Path(filename)
        tmp = target.with_suffix(target.suffix + ".tmp")
        logger.info(
            "Saving checkpoint | file={} | epoch={} | val_loss={:.4f} | val_f1_macro={:.4f} | run_id={}",
            target,
            epoch + 1,
            val_loss,
            val_f1_macro,
            self.run_id,
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_f1_macro": val_f1_macro,
            "run_id": self.run_id,
            "config": cfg_snapshot,
        }, tmp)

        tmp.replace(target)

        logger.success("Checkpoint saved locally at {}", target)

    def upload_checkpoint_to_azure(self,filename:str| Path="best.pt", blob_path:Optional[str|Path]=None)->str| None:
        ''' Uploads the checkpoint file from local storage to Azure Blob Storage.  
        Args:
            filename: The name of the checkpoint file to upload.
            blob_path: The path to the blob in Azure Blob Storage where the checkpoint will be uploaded.

        Returns:
            str: The path to the uploaded blob in Azure Blob Storage.
        '''

        if not self.run_id:
            raise RuntimeError("run_id must be set before uploading checkpoint.")
        
        blob_name = str(Path(blob_path)) if blob_path else self._blob_path(filename)
        local_file = self.checkpoint_dir / Path(filename)
        logger.info(
            "Uploading checkpoint | local_file={} | blob_name={} | container={}",
            local_file,
            blob_name,
            self.az_container,
        )

        try:

            with AzureBlobStorage(container_name=self.az_container,logger_config=load_yaml(log_file),log_type="ml") as az_:
                # upload local checkpoint file to blob storage under a run-specific path
             
                path = az_.upload_blob(blob_name=blob_name, 
                                       data=local_file,#gets locally
                                         Overwrite=True)
                
           

        except Exception as e:
            logger.exception(f"Error uploading checkpoint to Azure Blob Storage for {blob_name}: {e}")  
            return None
        
        if path is None:
            logger.error(f"Upload to {blob_name} failed (see AzureBlobStorage logs above).")
            return None
        
        logger.success("Checkpoint uploaded to Azure Blob Storage at {} in container {}", blob_name, self.az_container)
        return path

 

    def load_checkpoint(self, model, optimizer,scheduler, device,save_filename:str|Path="best.pt", blob_path:Optional[str|Path]=None)-> CheckpointState:
        ''' Loads checkpoint to local storage from Azure Blob Storage if it exists, and then loads the model 
        and optimizer state from the checkpoint. 

        Args:
            model: The model to load the checkpoint into.
            optimizer: The optimizer to load the checkpoint into.
            scheduler: The scheduler to load the checkpoint into.
            device: The device to map the model and optimizer to.
            save_filename: The name of the checkpoint file to save.
            blob_name: The name of the blob in Azure Blob Storage to load the checkpoint from.

        Returns:
            CheckpointState: A dataclass containing the start epoch, best validation loss, and resume run ID.

        '''

        if not self.resume_enabled:
            logger.info("Resume is disabled. Starting training from scratch.")
            return CheckpointState(start_epoch=0, best_val_loss=float('inf'), resume_run_id=None)
        
        resolved_blob_name = str(Path(blob_path)) if blob_path else self._blob_path("best.pt") #use filename argument else create
        local_path = self.checkpoint_dir / Path(save_filename)
        logger.warning(
            "Attempting to resume checkpoint | blob_name={} | local_path={} | container={}",
            resolved_blob_name,
            local_path,
            self.az_container,
        )

        try:
            with AzureBlobStorage(container_name=self.az_container, logger_config=load_yaml(log_file), log_type="ml") as az_loader:
                downloaded_url = az_loader.load_blob(blob_name=resolved_blob_name, local_path=local_path)
        except Exception as e:
            logger.exception(f"Unexpected error contacting Azure Blob Storage: {e}")
            return self._handle_missing()

        # load_blob swallows its own errors and returns None on failure
        if downloaded_url is None:
            logger.warning(f"Checkpoint blob not found or failed to download: {resolved_blob_name}")
            return self._handle_missing()

        logger.success("Checkpoint downloaded from Azure Blob Storage to {}", local_path)

        try:
            ckpt = torch.load(local_path, map_location=device)
        except Exception as e:
            logger.exception(f"Downloaded checkpoint file is corrupt or unreadable: {e}")
            return self._handle_missing()

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        resume_run_id = ckpt.get("run_id")

        logger.success(
            "Checkpoint loaded successfully. Resuming from epoch {} with best validation loss {}.",
            start_epoch,
            best_val_loss,
        )
        return CheckpointState(start_epoch=start_epoch, best_val_loss=best_val_loss, resume_run_id=resume_run_id)
        
    # Internal class helpers
    def _blob_path(self, filename: str | Path) -> str:
        """Single source of truth for the blob path — used by both upload and load
        so the two can never diverge."""
        if not self.run_id:
            raise RuntimeError("run_id must be set before building a blob path.")
        return str(Path(self.run_id) / self.checkpoint_dir_azure / Path(filename))
    
    def _handle_missing(self) -> CheckpointState:
        if self.resume_hard:
            raise RuntimeError(
                "resume_from_checkpoint=True and checkpoint_resume_hard=True "
                "but no checkpoint found. Exiting."
            )
        logger.warning("No checkpoint found. Starting fresh training with a clean state.")
        return CheckpointState(start_epoch=0, best_val_loss=float("inf"), resume_run_id=None)




