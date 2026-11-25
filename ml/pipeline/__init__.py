from __future__ import annotations
import click
from src.cli.utils import LazyPipelineGroup
from ml.cli import cli


command_modules = {
        "train": ("ml.pipeline.train", "train"),
        "evaluate": ("ml.pipeline.eval", "evaluate"),
        "preprocess": ("ml.pipeline.preprocess", "preprocess"),
        "load-metrics": ("ml.pipeline.mlflow", "load_metrics"),
        }

class PipelineGroup(LazyPipelineGroup):
    command_modules = command_modules


@cli.group(cls=PipelineGroup)
@click.pass_context
def pipeline(ctx):
    """ML pipeline commands.

    example:\n
        galax pipeline preprocess --run_preprocessing True \
        --preprocess_type classic
    """

    from src.utils.bootstrap import load_context

    ctx.obj = load_context(service="ml",std_out=False)  # Load ML context for pipeline commands
