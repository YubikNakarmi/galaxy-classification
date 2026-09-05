
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

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
    
    

    
    
    