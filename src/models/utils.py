from src.utils.azure_utils import AzureBlobStorage


class CheckpointManager:
    def __init__(self, container_name, checkpoint_dir_blob, checkpoint_dir_local):
        self.blob_helper = AzureBlobStorage(container_name=container_name)
        self.checkpoint_dir_blob = checkpoint_dir_blob
        self.checkpoint_dir_local = checkpoint_dir_local

    def save_checkpoint(self, local_checkpoint_path):
        """Save a checkpoint to Azure Blob Storage."""
        self.blob_helper.upload_blob(local_path=local_checkpoint_path, blob_name=self.checkpoint_dir_blob)

    def load_checkpoint(self, local_checkpoint_path):
        """Load a checkpoint from Azure Blob Storage."""
        self.blob_helper.load_blob(blob_name=self.checkpoint_dir_blob, local_path=local_checkpoint_path)