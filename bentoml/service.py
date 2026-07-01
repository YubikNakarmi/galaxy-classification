import mlflow
import bentoml
from mlflow.tracking import MlflowClient
from src.utils.yaml_utils import load_yaml
from pathlib import Path
from dotenv import load_dotenv
import os
from .runners import load_best_model
from PIL import Image
from runners import run_default_model,run_dirichlet_model,run_hubble_model,build_gradcam
from consts import DEVICE, experiment_root_default, experiment_root_hubble, experiment_root_dirichlet


root = Path(__file__).parent.parent

load_dotenv(root / '.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@bentoml.service(resources={"cpu": "2","memory": "4Gi"},
                 traffic={"max_concurrency": 5})
class GalaxyClassification:
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        self.default_model = load_best_model(experiment_name=experiment_root_default).eval().to(DEVICE)
        self.hubble_model = load_best_model(experiment_name=experiment_root_hubble).eval().to(DEVICE)
        self.dirichlet_model = load_best_model(experiment_name=experiment_root_dirichlet).eval().to(DEVICE)
        self.gradcam_default = build_gradcam(self.default_model)
        self.gradcam_hubble = build_gradcam(self.hubble_model)

    @bentoml.api(route = "/default",batchable=True, max_batch_size=16, max_latency_ms=300)
    def classify_default(self,images:list[Image.Image]):
        response = run_default_model(model = self.default_model,cam=self.gradcam_default,orig_img=images)
        return response




    