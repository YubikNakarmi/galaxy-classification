#update cached models, like promotion to champion
# check checksum, update metadata, etc
from __future__ import annotations
from ml.cache import cache
import click
import uuid
from pathlib import Path
from typing import Optional

root = Path(__file__).parent.parent.parent





@cache.command(name="import")
@click.pass_context

@click.option("--checkpoint_path", default = None, type=str, required=True, help="Path to the checkpoint file")
@click.option('--model_name', type=str, required=True, help='Registered model name to import')
@click.option('--variant', type=str, required=True, help='Variant name to import')
@click.option("--num_classes", default = 5,type=int, required=True, help="Number of classes in the model")
@click.option("--run_id", default = f"manual_{uuid.uuid4()}",type=str, required=False, help="Run ID for the model")
@click.option("--metric_name", default = "vf1_macro",type=str, required=False, help="Metric name to extract from checkpoint")
@click.option("--metric", default = None,type=float, required=False, help="Metric value to use if not present in checkpoint")
@click.option("--local_cache_dir", default = root / "models" / "cached",type=str, required=False, 
              help="Directory to store cached models")
@click.option("--metadata_save_file_path", default = root / "models" / "cached" / "model.json",type=str, 
              required=False, help="Path to save the metadata JSON file")

def import_model(ctx,variant:str, 
                 checkpoint_path:str, 
                 num_classes:int, 
                 run_id:str, 
                 metric_name:str,
                 metric:float,
                 model_name:str,
                 local_cache_dir:str,
                 metadata_save_file_path:str):
    
    from src.utils.cached_models import ModelLoader
    from src.cli.style import UI
    from src.utils.schemas import CandidateMetadata
    from loguru import logger
    from src.utils.bootstrap import load_context
    import datetime
    import torch


    contxt= ctx.obj or load_context(service="ml",std_out=False)  # Load ML context for pipeline commands and type hint
    run_id = run_id 
    ui= ctx.find_object(UI) or UI(verbose=0)
    cfg = contxt.cfg

    VARS = ["Default", "Hubble", "Dirichlet"]

    #path validators need to add
    
    if variant not in VARS:

        ui.error(f"Invalid variant '{variant}'. Must be one of {VARS}.")
        logger.error(f"Invalid variant '{variant}'. Must be one of {VARS}.")
        raise ValueError(f"Invalid variant '{variant}'. Must be one of {VARS}.")
    
    try:
    
        loader = ModelLoader(
            registered_model_name=model_name, ctx=contxt,
            local_cache=True, local_cache_dir=local_cache_dir,
            metadata_save_file_path = metadata_save_file_path,
            num_classes=num_classes, cache_metadata=True,
            variant=variant, metric=metric_name, checksum=True,
        )
        
        #checkpoint format and extract state_dict and metric if present
        state_dict, extracted_metric = _extract_checkpoint_info(checkpoint_path, device= cfg.env.device) 

        if metric is None:
            if extracted_metric is None:
                logger.error(
                    f"No '{metric_name}' found in checkpoint and none supplied via val_metric — ")
                ui.error(
                    f"No '{metric_name}' found in checkpoint and none supplied via val_metric — ")
                raise ValueError(
                    f"No '{metric_name}' found in checkpoint and none supplied via val_metric — "
                    "required for ranking against existing candidates."
                )
            metric = extracted_metric

        test_model = loader._init_model()
        test_model.load_state_dict(state_dict, strict=True) #map strictly to the skeleton

        cache_path = loader._cache_path(run_id)
        torch.save(state_dict, cache_path)
        logger.debug(f"Saved model state dict to cache: {cache_path}")

        checksum = loader.compute_checksum(run_id)
        loader._save_checksum(run_id, checksum)

        cand = CandidateMetadata(
            rank=0,
            run_id=run_id,
            model_name=model_name,
            experiment_name=model_name,
            metric={metric_name: metric},
            num_classes=num_classes,
            cached_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            save_path=str(loader.local_cache_dir),
            other_metrics={},
        )
        
        loader.metadata.update_metadata(cand, metric=metric_name, variant=variant, top_n=loader.top_n)
        logger.success(f"Imported checkpoint as run_id={run_id}, {metric_name}={metric}")
        ui.success(f"Imported checkpoint as run_id={run_id}, {metric_name}={metric}")

    except Exception as e:
        logger.error(f"Failed to import checkpoint: {e}")
        ui.error(f"Failed to import checkpoint: {e}",exec=True)
        raise 


def _extract_checkpoint_info(checkpoint_path: str | Path, device) -> tuple[dict, Optional[float]]:
    """ 
    Used to extract the model state_dict and optional validation metric from a checkpoint file.
    The checkpoint can be either a raw state_dict (models/cached format) or a training checkpoint.py 
    format (dict with 'model_state_dict' and optional 'val_f1_macro').

    Args:
        checkpoint_path (str | Path): Path to the checkpoint file.
        device: The device to load the checkpoint on.
    
    Returns: 
        tuple[dict, Optional[float]]: The extracted state dictionary and optional validation metric.
    """
    import torch

    obj = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        # already a bare state_dict — models/cached format, no metric available
        return obj, None

    if isinstance(obj, dict) and "model_state_dict" in obj:
        # training checkpoint.py format
        return obj["model_state_dict"], obj.get("val_f1_macro")

    raise ValueError(
        f"Unrecognized checkpoint format at {checkpoint_path}: "
        f"expected raw state_dict or dict with 'model_state_dict' key, "
        f"got keys={list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
    )

@cache.command()
@click.pass_context
def modify(ctx):
    pass