from pytorch_grad_cam import GradCAM
from src.utils.logging_config import setup_app_logging
from pytorch_grad_cam.utils.image import show_cam_on_image
from mlflow.tracking import MlflowClient
import mlflow
from src.utils.yaml_utils import load_yaml
from loguru import logger
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
import base64
from .schemas import ClassificationResult
from prometheus_client import Counter, Histogram
import time

from .consts import TRANSFORM, DEVICE, DEFAULT_CLASS_NAMES

root = Path(__file__).parent.parent
file_config = root / 'configs' / 'core.yaml'
log_yaml = load_yaml(file_config)
setup_app_logging(log_yaml)



ml_config = load_yaml(root / 'configs' / 'ml.yaml')


def load_best_model(experiment_name: str) -> torch.nn.Module:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_f1_macro DESC"],
        max_results=1,
    )

    if not runs:
        logger.error(f"No finished runs in experiment '{experiment_name}'")
        raise ValueError(f"No finished runs in experiment '{experiment_name}'")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    return mlflow.pytorch.load_model(model_uri) #preload while return

def build_gradcam(model: torch.nn.Module) -> GradCAM:

    target_layer = [model.group5.units[-1].conv2]#second last layer or last group to capture high feature maps
    return GradCAM(model=model, target_layers=target_layer)

def run_gradcam(
    cam: GradCAM,tensor: torch.Tensor,
    orig_img: Image.Image) -> str:
    
    """Return base64-encoded PNG heatmap overlay for a single image tensor.
    
    Args:
        cam: GradCAM instance for generating heatmaps.
        tensor: Preprocessed/transformed image tensor (C, H, W) for which to generate the heatmap.
        orig_img: Original PIL Image corresponding to the tensor, used for overlaying the heatmap

    Returns:        
        Base64-encoded PNG string of the heatmap overlay.
    
    """

    grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=None)[0]
    img_rgb = np.array(orig_img.resize((128, 128))) / 255.0
    heatmap = show_cam_on_image(img_rgb.astype(np.float32), grayscale_cam, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(heatmap).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()



def run_default_model(model: torch.nn.Module, cam: GradCAM, orig_img: list[Image.Image]) -> list[ClassificationResult]:
    
    tensor = torch.stack([TRANSFORM(img) for img in orig_img]).to(DEVICE)

    with torch.no_grad():
        logits  = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    results = []

    for i,(prob, img) in enumerate(zip(probabilities, orig_img)): #enumerate over each image and its corresponding probabilities
        pred_class = np.argmax(prob)
        heat_start = time.perf_counter()
        heatmap_b64 = run_gradcam(cam, tensor[i], img)
        heat_end = time.perf_counter()
        
        results.append(ClassificationResult(
                object_type = DEFAULT_CLASS_NAMES[pred_class],
                confidence_scores = float(prob[pred_class]),
                heatmap_b64 = heatmap_b64,
                heatmap_latency = heat_end - heat_start
        ))
    return results

def run_hubble_model():
    pass

def run_dirichlet_model():
    pass 