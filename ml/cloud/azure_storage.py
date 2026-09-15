import click
from ml.cloud import cloud
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

@cloud.command()
@click.pass_context
def check_storage_status(ctx):
    
    
    
    from dotenv import load_dotenv
    import os
    from src.cli.style import UI
    from src.utils.bootstrap import load_context
    from src.utils.azure_utils import AzureBlobStorage
    from rich.tree import Tree
    from rich.panel import Panel
    
    load_dotenv(ROOT / ".env")
    ui = ctx.find_object(UI) or UI(verbose=1)
    app_ctx = ctx.obj or load_context(service="ml",std_out=False) 

    azblob = os.getenv("AZURE_STORAGE_CONNECTION_STRING", None)
    if not azblob:
        ui.error("AZURE_STORAGE_CONNECTION_STRING is not set in the environment variables.")
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is not set in the environment variables.")

    azblob_client = AzureBlobStorage(storage_connection_string=azblob)
    info = azblob_client.print_info()
    
    root = Tree("Azure Storage Status", guide_style="bold bright_blue")
    
    for key, value in info.items():
        
        if isinstance(value, list):
            sub_tree = Tree(f"{key}", guide_style="dim")
            for val in value:
                sub_tree.add(f"{val}")
            root.add(sub_tree)
        else:
            root.add(f"{key}: {value}")


    ui.print(Panel(root, title="Azure Storage Status", border_style="blue"))

@cloud.command()
@click.option("--container_name",required=False, prompt="Container name",
              help="Name of the Azure Blob Storage container.")
@click.option("--blob_name","-b",required=True, prompt="Blob name",
              help="Name of the blob to upload.")
@click.option("--file_path","-f",required=True, prompt="File path", 
              help="Path to the file to upload.")
@click.pass_context
def upload_blob(ctx, blob_name: str, 
                file_path: str, 
                container_name: str=None):
    
    """Upload a blob to Azure Blob Storage."""
    from src.utils.azure_utils import AzureBlobStorage
    import os
    from dotenv import load_dotenv
    from src.cli.style import UI
    from src.utils.bootstrap import load_context
    from src.utils.validation import validate_path_exists
    
    
    
    load_dotenv(ROOT / ".env")
    ui = ctx.find_object(UI) or UI(verbose=1)
    app_ctx = ctx.obj or load_context(service="ml",std_out=False) 
    
    if not validate_path_exists([file_path],kind="file"):
        ui.error("The specified file path does not exist.")
        raise 

    azblob = os.getenv("AZURE_STORAGE_CONNECTION_STRING", None)
    if not azblob:
        ui.error("AZURE_STORAGE_CONNECTION_STRING is not set in the environment variables.")
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING is not set in the environment variables.")
    
    if not container_name:
        ui.warning("Container name not provided. Using default container from configuration.")
        container_name = app_ctx.core_cfg["cloud"]["azure"]["storage"]["default_container"]
    
    
    with ui.status(f"Uploading {file_path} to container '{container_name}' as blob '{blob_name}'..."):
        try:
            azblob_client = AzureBlobStorage(azblob)
            ui.debug(f"Uploading file '{file_path}' to container '{container_name}' with blob name '{blob_name}'.")
            azblob_client.upload_blob(container_name=container_name, 
                                       blob_name=blob_name, 
                                       file_path=file_path)
            ui.success(f"Successfully uploaded {file_path} to container '{container_name}' as blob '{blob_name}'.")
        except Exception as e:
            ui.error(f"Failed to upload blob: {e}", show_execption=True)
            raise e
    
    

def upload_blob_batch():
    pass
