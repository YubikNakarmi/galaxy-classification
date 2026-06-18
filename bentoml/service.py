import mlflow
import bentoml
from mlflow.tracking import MlflowClient
from src.utils.yaml_utils import load_yaml
from pathlib import Path
from dotenv import load_dotenv
import os
from .runners import load_best_model
from PIL import Image
from runners import run_default_model,run_dirichlet_model,run_grad_cam,run_hubble_model


root = Path(__file__).parent.parent

load_dotenv(root / '.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]
experiment_root_default = ml_yaml["variants"][ml_yaml["Default"]]["experiment"]["root"] #MLFLOW experiment name for a specific variant
experiment_root_hubble = ml_yaml["variants"][ml_yaml["Hubble"]]["experiment"]["root"]
experiment_root_dirichlet = ml_yaml["variants"][ml_yaml["Dirichlet:"]]["experiment"]["root"]
device = ml_yaml["env"]["device"]





@bentoml.service(resources={"cpu": "2","memory": "4Gi"},
                 traffic={"max_concurrency": 5})
class GalaxyClassification:
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        self.default_model = load_best_model(experiment_name=experiment_root_default).eval().to(device)
        self.hubble_model = load_best_model(experiment_name=experiment_root_hubble).eval().to(device)
        self.dirichlet_model = load_best_model(experiment_name=experiment_root_dirichlet).eval().to(device)

    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=300)
    def classify_default(self,images:list[Image.Image]):
        pass

    def classify_hubble(self,images:list[Image.Image]):
        pass

    def classify_dirichlet(self,images:list[Image.Image]):
        pass        

        


    