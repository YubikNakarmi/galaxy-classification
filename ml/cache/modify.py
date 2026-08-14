#update cached models, like promotion to champion
# check checksum, update metadata, etc

from ml.cache import cache
import click
import uuid
from pathlib import Path

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
