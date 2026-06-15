import mlflow
import bentoml
from mlflow.tracking import MlflowClient
from src.utils.yaml_utils import load_yaml
from pathlib import Path
from dotenv import load_dotenv
import os
from .runners import best_modes_default

root = Path(__file__).parent.parent

load_dotenv(root / '.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")



file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]
experiment_root = ml_yaml["variants"][ml_yaml["active_variant"]]["experiment"]["root"] #MLFLOW experiment name for a specific variant






@bentoml.service(resources={"cpu": "2","memory": "4Gi"},
                 traffic={"max_concurrency": 5})
class GalaxyClassification:
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        


    