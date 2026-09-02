
from ml import config
from src.cli.style import UI
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

def check_azure_storage(ui:Optional[UI] = None)-> bool:
    
    from src.utils.bootstrap import load_context
    
    """
    Check if the Azure Storage connection string is set in the environment variables.
    """
    ui = ui or UI()
    ctx = load_context()
    cfg = ctx.cfg
    
    dvc_container = cfg.storage.cloud.dvc.container
    experiments_container = cfg.storage.cloud.experiments.container
    
    