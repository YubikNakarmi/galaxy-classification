import mlflow
import bentoml
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from PIL import Image
from prometheus_client import Counter, Histogram
import time

from src.utils.cached_models import ModelLoader
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import intercept_stdlib_logging, setup_app_logging
from bentoml_api.runners import run_default_model, run_dirichlet_model, run_hubble_model, build_gradcam
from src.utils.cli_bootstrap import load_context

from bentoml_api.consts import TRANSFORM, DEVICE, DEFAULT_CLASS_NAMES,\
default_model_name, hubble_model_name, dirichlet_model_name


from loguru import logger

root = Path(__file__).parent.parent
load_dotenv(root / '.env')
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# loggings (no down stream setup)
file_config = root / 'configs' / 'core.yaml'
log_yaml = load_yaml(file_config)
log_json = os.getenv("LOG_JSON", "True").lower() in ("true", "1", "yes")

setup_app_logging(log_yaml,service = "Bentoml",json_logs = log_json)
intercept_stdlib_logging() # logs stdlib to loguru in container logs

# Prometheus metrics
h = Histogram('request_latency_seconds', 'Request latency in seconds', ['endpoint'])
h_gradcam = Histogram('gradcam_latency_seconds', 'GradCAM latency in seconds', ['endpoint'])
number_requests = Counter('number_of_requests', 'Number of requests received', ['endpoint'])

@bentoml.service(resources={"cpu": "2","memory": "4Gi"},
                 traffic={"max_concurrency": 5})

class GalaxyClassification:
    
    def __init__(self):
        ctx = load_context(service="bentoml")
        cfg = ctx.cfg
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        modelloader_default = ModelLoader(registered_model_name=default_model_name,
                                          local_cache=True, local_cache_dir=root / "models/cached",
                                          cfg = cfg,
                                          model_top_n = 3,metric = "f1_macro")

        def_model= modelloader_default.load_champion_model() 

        self.default_model = def_model if def_model else \
        modelloader_default.load_with_fallback().eval().to(DEVICE)#change to "eval_f1_macro" on production
        

        #self.hubble_model = load_best_model(experiment_name=experiment_root_hubble).eval().to(DEVICE)
        #self.dirichlet_model = load_best_model(experiment_name=experiment_root_dirichlet).eval().to(DEVICE)
        self.gradcam_default = build_gradcam(self.default_model)
        #self.gradcam_hubble = build_gradcam(self.hubble_model)
        logger.bind(component = "startup").info("GalaxyClassification service initialized successfully")

    @bentoml.api(route = "/default",batchable=True, max_batch_size=16, max_latency_ms=300)
    def classify_default(self,images:list[Image.Image]):


        start = time.perf_counter()
        response = run_default_model(model = self.default_model,cam=self.gradcam_default,orig_img=images)

        end = time.perf_counter()
        h.labels(endpoint="/default").observe(end - start)

        logger.bind(endpoint="/default",ms =1 ).info(f"Processed {len(images)} images in {end - start:.4f} seconds")
        number_requests.labels(endpoint="/default").inc()

        if response:

            #Unpack the response to get the heatmap latencies
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


    