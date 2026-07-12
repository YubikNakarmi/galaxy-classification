import mlflow
import bentoml
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from PIL import Image
from prometheus_client import Counter, Histogram
import time

root = Path(__file__).parent.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.utils.yaml_utils import load_yaml
from bentoml_api.runners import load_best_model
from bentoml_api.runners import run_default_model, run_dirichlet_model, run_hubble_model, build_gradcam
from bentoml_api.consts import DEVICE, experiment_root_default, experiment_root_hubble, experiment_root_dirichlet

load_dotenv(root / '.env')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

h = Histogram('request_latency_seconds', 'Request latency in seconds', ['endpoint'])
h_gradcam = Histogram('gradcam_latency_seconds', 'GradCAM latency in seconds', ['endpoint'])
number_requests = Counter('number_of_requests', 'Number of requests received', ['endpoint'])

@bentoml.service(resources={"cpu": "2","memory": "4Gi"},
                 traffic={"max_concurrency": 5})
class GalaxyClassification:
    
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        self.default_model = load_best_model(experiment_name=experiment_root_default).to(DEVICE)
        #self.hubble_model = load_best_model(experiment_name=experiment_root_hubble).eval().to(DEVICE)
        #self.dirichlet_model = load_best_model(experiment_name=experiment_root_dirichlet).eval().to(DEVICE)
        self.gradcam_default = build_gradcam(self.default_model)
        #self.gradcam_hubble = build_gradcam(self.hubble_model)

    @bentoml.api(route = "/default",batchable=True, max_batch_size=16, max_latency_ms=300)
    def classify_default(self,images:list[Image.Image]):

        start = time.perf_counter()
        response = run_default_model(model = self.default_model,cam=self.gradcam_default,orig_img=images)

        end = time.perf_counter()
        h.labels(endpoint="/default").observe(end - start)
        number_requests.labels(endpoint="/default").inc()

        if response:
            heatmap_latencies = [lat for lat in (res.get("heatmap_latency") for res in response) if lat is not None]
            avg_heatmap_latency = (sum(heatmap_latencies) / len(heatmap_latencies)) if heatmap_latencies else 0
            h_gradcam.labels(endpoint="/default").observe(avg_heatmap_latency)
            
        else:
            avg_heatmap_latency = 0
            h_gradcam.labels(endpoint="/default").observe(0)
        return response
    
    @bentoml.api(route = "/hubble",batchable=True, max_batch_size=16, max_latency_ms=300)
    def classify_hubble(self,images:list[Image.Image]):
        start = time.perf_counter()

        #inferecne .....
        
        end = time.perf_counter()
        h.labels(endpoint="/hubble").observe(end - start)
        number_requests.labels(endpoint="/hubble").inc()
        pass

    @bentoml.api(route = "/dirichlet",batchable=True, max_batch_size=16, max_latency_ms=300)
    def classify_dirichlet(self,images:list[Image.Image]):
        start = time.perf_counter()
        # inference .....
        end = time.perf_counter()
        h.labels(endpoint="/dirichlet").observe(end - start)
        number_requests.labels(endpoint="/dirichlet").inc()
        pass


    