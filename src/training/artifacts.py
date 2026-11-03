''' Artifacts management for MLflow and Azure Blob Storage. 
This module provides functions to register models with MLflow, upload training logs, reports, and tensorboard artifacts to Azure Blob Storage. 
It also sets up logging configurations for both application and ML logging.'''
from __future__ import annotations

import click
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


import datetime
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Optional




if TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion
    from mlflow.tracking import MlflowClient
    from torch import Tensor
    from torch.nn import Module
 

root = Path(__file__).parent.parent.parent
load_dotenv(root / ".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

SkeletonFactory = Callable[[], "Module"] #callable skeleton of module for typing
_STATE_KEYS: Final[tuple[str, ...]] = ("model_state", "state_dict", "model_state_dict")



def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



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
 
def upload_artifact(blob: AzureBlobStorage, run_id: str,experiment_name: str,cfg)-> None:

    
    log_path = root / core_cfg["logging"]["ml"]["file"]
    logger.info("Uploading artifacts for run {} from {}", run_id, log_path)
 
    url = blob.upload_blob(blob_name=f"{run_id}/logs/training.log", data=log_path)
    mlflow.set_tag("log_url", url)
    logger.success("Uploaded training log to {}", url)
    
    #tensorboard
    try:
        report_path = cfg.storage.local.logs.reports_dir / f"{experiment_name}.md"
        logger.info("Uploading training report from {}", report_path)
        url = blob.upload_blob(blob_name=f"{run_id}/reports/report.md", data=report_path)
        mlflow.set_tag("report_url", url)
        logger.success("Uploaded training report to {}", url)

    except Exception as e:
        logger.exception(f"Could not upload report for run {run_id}: {e}")
    
    #logs
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

def get_champion(client, name, alias):
    try:
        return client.get_model_version_by_alias(name, alias)
    except mlflow.MlflowException as e:
        if e.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None
    
def should_promote(client, name, alias, metric_name, candidate_metric, min_improvement)->tuple[bool,str]:
    
    champ = get_champion(client, name, alias)
    if champ is None:
        return True, "no_champion"
    raw = champ.tags.get(metric_name)
    if raw is None:
        return True, "champion_missing_metric"
    if candidate_metric > float(raw) + min_improvement:
        return True, f"improved_{metric_name}"
    return False, "not_better"

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
    
def load_local_model(model_path: str | Path, build_skeleton: SkeletonFactory) -> Module:
    """Load a local checkpoint as an eval-mode nn.Module on CPU.

    Accepts an MLflow model directory, a fully pickled module, or a state_dict
    checkpoint (optionally nested under model_state / state_dict / model_state_dict).
    ``build_skeleton`` is only called for state_dict checkpoints.
    """
    import torch
    from loguru import logger

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model path {p} does not exist.")

    if p.is_dir():
        if not (p / "MLmodel").exists():
            raise ValueError(f"{p} is a directory but not an MLflow model (no MLmodel file).")
        import mlflow.pytorch

        loaded: Module = mlflow.pytorch.load_model(str(p))
        return loaded.eval()

    obj: Any
    try:
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning("weights_only load failed for {}; falling back to full unpickle.", p)
        obj = torch.load(p, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        return obj.eval()
    if not isinstance(obj, dict):
        raise TypeError(f"Unrecognised checkpoint type {type(obj).__name__} in {p}")

    raw_state: dict[str, Any] = next((obj[k] for k in _STATE_KEYS if k in obj), obj)
    state: dict[str, Tensor] = {k.removeprefix("module."): v for k, v in raw_state.items()}

    model = build_skeleton()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Checkpoint {p} doesn't match the model architecture:\n{e}") from e

    logger.info("Loaded state_dict from {} into skeleton model.", p)
    return model.eval()

def upload_model(
    model: Module,
    registered_name: str,
    source_path: Optional[str | Path] = None,) -> str:
    """Log ``model`` in its own run and register it. Returns the new version string.

    The run is closed before returning, so it never collides with the eval run.
    """
   

    with mlflow.start_run(run_name=f"upload_{registered_name}_{_timestamp()}") as run:
        info = mlflow.pytorch.log_model(
            model,
            name="model", 
            registered_model_name=registered_name,
        )
        mlflow.set_tags(
            {
                "registered_during_eval": "True",
                "recovered_from_checkpoint": "True" if source_path else "False",
                "source_path": str(source_path or ""),
            })
        

    version = getattr(info, "registered_model_version", None)
    if version is None: #older mlflow vers
        versions = MlflowClient().search_model_versions(f"run_id='{run.info.run_id}'")
        if not versions:
            raise click.ClickException("Model was logged but no registry version was found.")
        version = max(versions, key=lambda v: int(v.version)).version
    return str(version)