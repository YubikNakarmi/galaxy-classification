from src.utils.azure_utils import AzureBlobStorage
import torch
from pathlib import Path




class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, checkpoint_blob_path: str, resume_from_checkpoint: bool):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_blob_path = checkpoint_blob_path
        self.resume_from_checkpoint = resume_from_checkpoint

    def load_checkpoint(self, model, optimizer, scheduler, device):
        if self.resume_from_checkpoint and Path(self.checkpoint_dir).exists():
            blob_helper = AzureBlobStorage(container_name=checkpoint_container_name)
            blob_helper.load_blob(blob_name=self.checkpoint_blob_path, local_path=self.checkpoint_dir / "best.pt")
            ckpt = torch.load(self.checkpoint_dir / "best.pt", map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])