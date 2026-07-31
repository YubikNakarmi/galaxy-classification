import click
from .train import train
from .eval import evaluate

click.group()
def cli():
    """Command line interface for the Galaxy Classification project."""
    pass

cli.add_command(train)
cli.add_command(evaluate)