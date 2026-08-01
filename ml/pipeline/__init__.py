import click
from src.utils.cli_bootstrap import load_context
@click.group()
@click.pass_context
def pipeline(ctx):
    """ML pipeline commands.

    example:
    ``` shell
        galax pipline preprocess --run_preprocessing True --preprocess_type classic
        ```
    
    """

    ctx.obj = load_context(service="ml")