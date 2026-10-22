from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
import torch
from torchvision import transforms 
import random
from torchvision.transforms import functional as tf
from typing import Optional
from pathlib import Path

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



def load_mlflow_metrics(run_id: str, metrics: Optional[list[str]]=None, save_path:Optional[str|Path]=None) -> dict:
    """_summary_

    Args:
        run_id (str): run id to pull metrics from
        metrics (Optional[list[str]], optional): List of metric names to load. If None, all metrics will be loaded. Defaults to None.
        save_path (Optional[str | Path], optional): Path to save the metrics CSV files. Defaults to None.

    Returns:
        dict: all metrics dictionary
        
    Example:
    ```python
    metrics = load_mlflow_metrics(run_id="e3b866df7f01417fb0db72e0da051f23", metrics=["accuracy", "f1_macro"], save_path="metrics/")
    ```
    """
    
    import mlflow
    import pandas as pd

    # Load metrics from MLflow
    client = mlflow.tracking.MlflowClient()
    
    run = client.get_run(run_id)
    
    if metrics is None:
        metrics = list(run.data.metrics.keys()) #getall keys metrics
    
    metrics_data = {}
    for metric_name in metrics:
        metric_history = client.get_metric_history(run_id, metric_name)
        
        if metric_history:
            df = pd.DataFrame([{"step": m.step, "value": m.value} for m in metric_history])
            
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path / f"{metric_name}.csv", index=False)
                
            metrics_data[metric_name] = df
        else:
            df = pd.DataFrame(columns=["step", "value"])
            metrics_data[metric_name] = df
            
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path / f"{metric_name}.csv", index=False)

    return metrics_data