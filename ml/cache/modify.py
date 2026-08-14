#update cached models, like promotion to champion
# check checksum, update metadata, etc

from ml.cache import cache
import click
import uuid
from pathlib import Path
from typing import Optional

root = Path(__file__).parent.parent.parent


@cache.command()
@click.pass_context
def modify(ctx):
    pass


@cache.command()
@click.pass_context
@click.option("--checkpoint_path", default = None, type=str, required=True, help="Path to the checkpoint file")
@click.option('--variant', type=str, required=True, help='Variant name to import')
@click.option("--num_classes", default = 5,type=int, required=True, help="Number of classes in the model")
@click.option("--run_id", default = f"manual_{uuid.uuid4()}",type=str, required=False, help="Run ID for the model")
@click.option("--file_dir", default = root / "models" / "cached",type=str, required=False, 
              help="Directory where the model files are located")

def import_model(ctx):
    from src.utils.cached_models import ModelLoader
    import torch



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