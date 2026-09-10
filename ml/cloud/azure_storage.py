import click
from ml.cloud import cloud

@cloud.command()
@click.option("--container_name", prompt="Container name", help="Name of the Azure Blob Storage container.")
@click.option("--blob_name","-b", prompt="Blob name", help="Name of the blob to upload.")
@click.option("--file_path","-f", prompt="File path", help="Path to the file to upload.")
@click.pass_context
def upload_blob(ctx, container_name: str, blob_name: str, file_path: str):
    """Upload a file to Azure Blob Storage."""
    from src.utils.azure_utils import AzureBlobStorage
    
    
    

def upload_blob_batch():
    pass
