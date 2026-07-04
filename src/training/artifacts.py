

import torch
import mlflow
from dotenv import load_dotenv
from pathlib import Path
from torch.utils.data import DataLoader
import os
from loguru import logger
import datetime
from mlflow.models.signature import infer_signature


from src.utils.azure_utils import AzureBlobStorage
from src.utils.yaml_utils import load_yaml
from src.utils.logging_config import setup_ml_logging
from src.config import load_and_validate_config


root = Path(__file__).parent.parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000")

log_config = load_yaml(root / 'configs' / 'core.yaml')
setup_ml_logging(log_config)

core_cfg = load_yaml(root / 'configs' / 'core.yaml')

file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
cfg = load_and_validate_config(file_config)

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
    
    # getting model signatures and sample for pt2 serialization
    with torch.no_grad():
        preds_np = torch.softmax(model(sample_images), dim=1).cpu().numpy()
 
    signature = infer_signature(sample_images.cpu().numpy(), preds_np)
 
    mlflow.pytorch.log_model(
        model,
        serialization_format="pt2", #most robust format for torchscript and future compatibility
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
    logger.info("Uploaded training log to {}", url)
 
    try:
        report_path = cfg.storage.local.logs.reports_dir / f"{experiment_name}.md"
        logger.info("Uploading training report from {}", report_path)
        url = blob.upload_blob(blob_name=f"{run_id}/reports/report.md", data=report_path)
        mlflow.set_tag("report_url", url)
        logger.info("Uploaded training report to {}", url)

    except Exception as e:
        logger.exception(f"Could not upload report for run {run_id}: {e}")
 
    try:
        tb_run_dir = cfg.storage.local.logs.tensorboard_dir / experiment_name
        logger.info("Uploading tensorboard artifacts from {}", tb_run_dir)
        for file in tb_run_dir.iterdir():
            if file.is_file():
                url = blob.upload_blob(blob_name=f"{run_id}/tensorboard/{file.name}", data=file)
                mlflow.set_tag(f"tensorboard_{file.name}", url)
                logger.info("Uploaded tensorboard file {} to {}", file.name, url)

    except Exception as e:
        logger.exception(f"Could not upload tensorboard logs for run {run_id}: {e}")