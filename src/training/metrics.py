from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
import torch
from torchvision import transforms 
import random
from torchvision.transforms import functional as tf
from typing import Optional
from pathlib import Path
from src.cli.style import UI
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")


class RandomCenterCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        size = random.randint(self.min_size, self.max_size)
        return tf.center_crop(img, size)

def make_metrics(num_classes:int, device: torch.device) -> dict:
    return {
        "accuracy":           Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "f1_micro":           F1Score(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "f1_macro":           F1Score(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "f1_classwise":       F1Score(task="multiclass", average="none", num_classes=num_classes).to(device),
        "precision_micro":    Precision(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "precision_macro":    Precision(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "precision_classwise":Precision(task="multiclass", average="none", num_classes=num_classes).to(device),
        "recall_micro":       Recall(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "recall_macro":       Recall(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "recall_classwise":   Recall(task="multiclass", average="none", num_classes=num_classes).to(device),
        "roc_auc":            AUROC(task="multiclass", num_classes=num_classes).to(device),
    }
# Hardcoded as there will be less editing of these transforms, but can be moved to config if needed
MEAN = [0.1376, 0.1276, 0.1187]
STD  = [0.1728, 0.1586, 0.1488]

raw_transforms = transforms.Compose([
    # rotate the full image first; the crops below sit inside the valid region
    transforms.RandomRotation(degrees=[0, 360],
                              interpolation=transforms.InterpolationMode.BILINEAR),
    RandomCenterCrop(170, 240),
    transforms.Resize(160),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # no hue or saturation
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(164),   # = 0.8 × mean train crop (205)
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])



def load_mlflow_metrics(run_id: str, metrics: Optional[list[str]]=None,
                        save_path:Optional[str|Path]=None,
                        ui:UI= UI(verbose=0),
                        exclude_system: bool = True) -> dict:
    """
    Load metrics from MLflow for a given run.

    Args:
        run_id (str): run id to pull metrics from
        metrics (Optional[list[str]], optional): List of metric names to load. If None, all metrics will be loaded. Defaults to None.
        save_path (Optional[str | Path], optional): Directory or CSV path for the
            combined metrics file. Defaults to None.
        exclude_system (bool, optional): Whether to exclude system metrics. Defaults to True.

    Returns:
        dict: all metrics dictionary

    Example:
    ```python
    metrics = load_mlflow_metrics(run_id="e3b866df7f01417fb0db72e0da051f23", metrics=["accuracy", "f1_macro"], save_path="metrics/")
    ```
    """

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Load metrics from MLflow
    client = mlflow.tracking.MlflowClient()
    
    ui.debug(f"tracking set to {os.getenv('MLFLOW_TRACKING_URI')}")
    ui.info(f"Loading metrics for run_id: {run_id}...")
    run = client.get_run(run_id)
    
    if metrics is None:
        if exclude_system:
            metrics = [k for k in run.data.metrics.keys() if not k.startswith("system/")]
        else:
            metrics = list(run.data.metrics.keys()) #getall keys metrics
        ui.warning(f"No metrics specified. Loading all metrics: {metrics}")
    
    metrics_data = {}
    combined_rows = []
    for metric_name in metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        
        if metric_history:
            rows = [
                {
                    "metric": metric_name,
                    "step": m.step,
                    "value": m.value,
                    "timestamp": m.timestamp,}
                for m in metric_history
            ]
            combined_rows.extend(rows)
            
            df = pd.DataFrame(rows).drop(columns=["metric"])
            metrics_data[metric_name] = df
            ui.success(f"Metrics loaded for {metric_name}")
        else:

            ui.warning(f"No data found for metric: {metric_name}. Creating empty DataFrame.")
            
            df = pd.DataFrame(columns=["step", "value", "timestamp"])
            metrics_data[metric_name] = df
            ui.success(f"Metrics loaded for {metric_name}")

    if save_path:
        save_path = Path(save_path)
        
        output_path = save_path if save_path.suffix.lower() == ".csv" else save_path / "metrics.csv"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        combined_df = pd.DataFrame(
            combined_rows,
            columns=["metric", "step", "value", "timestamp"],
        )
        combined_df.to_csv(output_path, index=False)
        ui.success(f"Metrics saved to {output_path}")

    return metrics_data
