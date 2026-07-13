from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_app_logging, intercept_stdlib_logging

from loguru import logger
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
import base64
import time
from typing import Optional
import os
from bentoml_api.schemas import ClassificationResult

from bentoml_api.consts import TRANSFORM, DEVICE, DEFAULT_CLASS_NAMES,\
experiment_root_default, experiment_root_hubble, experiment_root_dirichlet

root = Path(__file__).parent.parent
file_config = root / 'configs' / 'core.yaml'
log_yaml = load_yaml(file_config)
log_json = os.getenv("LOG_JSON", "True").lower() in ("true", "1", "yes")


setup_app_logging(log_yaml,service = "Bentoml",json_logs = log_json)
intercept_stdlib_logging() # logs stdlib to loguru in container logs


def build_gradcam(model: torch.nn.Module) -> Optional[GradCAM]:
    
    try:
        target_layer = [model.group5.units[-1].conv2]#second last layer or last group to capture high feature maps
        return GradCAM(model=model, target_layers=target_layer)
    
    except Exception as e:
        logger.critical(f"Failed to build GradCAM: {e}")
        return None
    
def run_gradcam(
    cam: GradCAM,tensor: torch.Tensor,
    orig_img: Image.Image) -> Optional[str]:
    
    """Return base64-encoded PNG heatmap overlay for a single image tensor.
    
    Args:
        cam: GradCAM instance for generating heatmaps.
        tensor: Preprocessed/transformed image tensor (C, H, W) for which to generate the heatmap.
        orig_img: Original PIL Image corresponding to the tensor, used for overlaying the heatmap

    Returns:        
        Base64-encoded PNG string of the heatmap overlay.
    
    """

    logger.bind(component = "gradcam").info("Generating GradCAM heatmap")
    try:

        grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=None)[0]
        img_rgb = np.array(orig_img.resize((128, 128))) / 255.0
        heatmap = show_cam_on_image(img_rgb.astype(np.float32), grayscale_cam, use_rgb=True)
        buf = io.BytesIO()
        Image.fromarray(heatmap).save(buf, format="PNG")

        logger.success("GradCAM heatmap generated successfully")
        return base64.b64encode(buf.getvalue()).decode()
    
    except Exception as e:
        logger.critical(f"Failed to generate GradCAM heatmap: {e}")
        return None



def run_default_model(model: torch.nn.Module, cam: GradCAM, orig_img: list[Image.Image]) -> list[ClassificationResult]:
    
    logger.bind(variant = "default").info("Running inference for default model")
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
                object_type = str(DEFAULT_CLASS_NAMES[pred_class]),
                confidence_scores = float(prob[pred_class]),
                heatmap_b64 = heatmap_b64,
                heatmap_latency = heat_end - heat_start
        ))
    
    return results

def run_hubble_model():
    pass

def run_dirichlet_model():
    pass 


