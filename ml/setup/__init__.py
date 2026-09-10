from __future__ import annotations
import click 
import importlib


from ml.cli import cli
from src.cli.utils import LazyPipelineGroup

class LazySetupGroup(LazyPipelineGroup):

    command_modules = {
        "setup": ("ml.setup.setup", "main"),
        }

    

@cli.group(cls=LazySetupGroup)
@click.pass_context
def setup(ctx):
    '''Setup commands for the Galaxy Classification project.'''
    from src.utils.bootstrap import load_context
    
    ctx.obj = load_context(service="cli",std_out=False)  # Load ML context for setup commands
    