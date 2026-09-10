import click
from ml.cli import cli
from src.cli.utils import LazyPipelineGroup

command_modules = {
    "azure_storage": ("ml.cloud.azure_storage", "cloud"),
}

class CloudGroup(LazyPipelineGroup):
    command_modules = command_modules




@cli.group(cls = CloudGroup)
@click.pass_context
def cloud(ctx):
    """Cloud storage commands."""
    from src.utils.bootstrap import load_context

    load_context(service="ml",std_out=False)  # Load ML context for pipeline commands
