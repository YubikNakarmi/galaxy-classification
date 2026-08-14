''' Artifacts management for MLflow and Azure Blob Storage. 
This module provides functions to register models with MLflow, upload training logs, reports, and tensorboard artifacts to Azure Blob Storage. 
It also sets up logging configurations for both application and ML logging.'''

import torch
import mlflow
from dotenv import load_dotenv
from pathlib import Path
from torch.utils.data import DataLoader
import os
from loguru import logger
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.utils.azure_utils import AzureBlobStorage
from src.utils.yaml_utils import load_yaml
from src.config import load_and_validate_config


root = Path(__file__).parent.parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config)
core_cfg = load_yaml(root / 'configs' / 'core.yaml')

#used throughout all loggings runs for local saves, cloud saves use mlflow run_id
# normalize device to a torch.device once
dev = torch.device(cfg.env.device)
active_variant = cfg.active_variant



def register_model(model,val_loader:DataLoader):
    sample_images, _ = next(iter(val_loader))
    sample_images = sample_images.to(dev)
    logger.info(
        "Registering model with MLflow | model_name={} | input_shape={} | device={}",
        cfg.variant.mlflow.model_name,
        tuple(sample_images.shape),
        dev,
    )
    
    # getting model signatures and sample 
    with torch.no_grad():
        preds_np = torch.softmax(model(sample_images), dim=1).cpu().numpy()
 
    signature = infer_signature(sample_images.cpu().numpy(), preds_np)
 
    mlflow.pytorch.log_model(
        model,
        #serialization_format="pt2", using non pt2 for better grandcam compatibility
        input_example=sample_images.cpu(),
        name="model",
        registered_model_name=cfg.variant.mlflow.model_name,
        signature=signature,
    )
    logger.success("Model logged to MLflow registry as {}.", cfg.variant.mlflow.model_name)
 
def upload_artifact(blob: AzureBlobStorage, run_id: str,experiment_name: str)-> None:

    
    log_path = root / core_cfg["logging"]["ml"]["file"]
    logger.info("Uploading artifacts for run {} from {}", run_id, log_path)
 
    url = blob.upload_blob(blob_name=f"{run_id}/logs/training.log", data=log_path)
    mlflow.set_tag("log_url", url)
    logger.success("Uploaded training log to {}", url)
 
    try:
        report_path = cfg.storage.local.logs.reports_dir / f"{experiment_name}.md"
        logger.info("Uploading training report from {}", report_path)
        url = blob.upload_blob(blob_name=f"{run_id}/reports/report.md", data=report_path)
        mlflow.set_tag("report_url", url)
        logger.success("Uploaded training report to {}", url)

    except Exception as e:
        logger.exception(f"Could not upload report for run {run_id}: {e}")
 
    try:
        tb_run_dir = cfg.storage.local.logs.tensorboard_dir / experiment_name
        logger.info("Uploading tensorboard artifacts from {}", tb_run_dir)
        for file in tb_run_dir.iterdir():
            if file.is_file():
                url = blob.upload_blob(blob_name=f"{run_id}/tensorboard/{file.name}", data=file)
                mlflow.set_tag(f"tensorboard_{file.name}", url)
                logger.success("Uploaded tensorboard file {} to {}", file.name, url)

    except Exception as e:
        logger.exception(f"Could not upload tensorboard logs for run {run_id}: {e}")


def promote_if_better(
    registered_model_name: str,
    candidate_version: str,
    candidate_metric: float,
    metric_name: str = "eval_f1_macro",
    alias: str = "champion",
    min_improvement: float = 0.005) -> bool:

    client = MlflowClient()

    try:
        #using alias for best model after eval
        current_champion = client.get_model_version_by_alias(registered_model_name, alias)
        current_metric_str = current_champion.tags.get(metric_name)
        current_metric = float(current_metric_str) if current_metric_str is not None else None    

    except:
        logger.warning("No current champion found for model {}. " \
        "Promoting candidate version {} as the new champion.", registered_model_name, candidate_version)

        current_champion=None
        current_metric = None

    if current_champion is None or current_metric is None:
        logger.info("No current champion found for model {}. Promoting candidate version {} as the new champion.", registered_model_name, candidate_version)
        client.set_registered_model_alias(registered_model_name, alias, candidate_version)
        client.set_registered_model_tag(name=registered_model_name, key=metric_name,value=str(candidate_metric)) 
        return True    

    logger.info("Comparing current champion version {} with metric {}={} to candidate version {} with metric {}={}", 
                current_champion.version, metric_name, current_metric, candidate_version, metric_name, candidate_metric)

    if candidate_metric > current_metric + min_improvement:
        client.set_registered_model_alias(registered_model_name ,alias, candidate_version)
        client.set_registered_model_tag(name=registered_model_name, key=metric_name,value=str(candidate_metric))
        logger.success("Current champion version {} is better than candidate version {}. No promotion.", current_champion.version, candidate_version)

        return True
    else:
        logger.info("Candidate version {} is not better than current champion version {}. No promotion.", candidate_version, current_champion.version)
        return False
    