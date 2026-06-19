from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from mlflow.tracking import MlflowClient
import mlflow

from src.utils.yaml_utils import load_yaml
from pathlib import Path
import torch
import loguru
import numpy as np
from PIL import Image
import io
import base64


root = Path(__file__).parent.parent


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]


def load_best_model(experiment_name: str) -> torch.nn.Module:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_f1_macro DESC"],
        max_results=1,
    )

    if not runs:
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
    
    """Return base64-encoded PNG heatmap overlay for a single image tensor."""

    grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=None)[0]
    img_rgb = np.array(orig_img.resize((128, 128))) / 255.0
    heatmap = show_cam_on_image(img_rgb.astype(np.float32), grayscale_cam, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(heatmap).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()



def run_default_model(cam: GradCAM, tensor: torch.Tensor, orig_img: Image.Image):
    pass
    

def run_hubble_model():
    pass

def run_dirichlet_model():
    pass 