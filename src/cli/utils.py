
from typing import Optional
from pathlib import Path
import click
import importlib


ROOT = Path(__file__).parent.parent.parent



class LazyPipelineGroup(click.Group):
    """Load pipeline commands only when they are invoked.
    
    Example:
    ```python
        command_modules = {
                "train": ("ml.pipeline.train", "train"),
                "evaluate": ("ml.pipeline.eval", "evaluate"),
                "preprocess": ("ml.pipeline.preprocess", "preprocess"),
                
            }
    ```
    """

    def __init__(self, command_modules: Optional[dict[str, tuple[str, str]]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs) #if you want to add more attributes to the class, you can do it here
        if command_modules is not None:
            self.command_modules = command_modules


    def list_commands(self, ctx):
        return sorted(self.command_modules)

    def get_command(self, ctx, name):
        module_info = self.command_modules.get(name)
        if module_info is None:
            return None

        module_name, command_name = module_info
        module = importlib.import_module(module_name)
        return getattr(module, command_name)

def check_azure_storage()-> dict[str,bool]:
    
    from src.utils.bootstrap import load_context
    from src.utils.azure_utils import CheckStatus
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    
    """
    Check if the Azure Storage connection string is set in the environment variables.
    """
    ctx = load_context()
    cfg = ctx.cfg
    
    dvc_container = cfg.storage.cloud.dvc.container
    experiments_container = cfg.storage.cloud.experiments.container
    
    _AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    st = CheckStatus(conn_string= _AZURE_STORAGE_CONNECTION_STRING)
    
    azure_status = st.check_connection()
    containers_status = st.check_containers([dvc_container, experiments_container])
    
    
    return {
        "azure": azure_status,
        "dvc": containers_status[dvc_container],
        "experiments": containers_status[experiments_container]
    }
    
    

    
    
    