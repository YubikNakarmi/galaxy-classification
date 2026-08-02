import click
from ml.pipeline import pipeline


@click.group()
def cli():
    """Command line interface for the Galaxy Classification project."""
    pass

cli.add_command(pipeline)
