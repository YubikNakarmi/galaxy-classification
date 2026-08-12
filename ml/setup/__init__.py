from __future__ import annotations
import click 
import importlib


from ml.cli import cli

class LazyPipelineGroup(click.Group):
    """Load pipeline commands only when they are invoked."""

    command_modules = {
        "setup": ("ml.setup.setup", "main"),
        
    }

    def list_commands(self, ctx):
        return sorted(self.command_modules)

    def get_command(self, ctx, name):
        module_info = self.command_modules.get(name)
        if module_info is None:
            return None

        module_name, command_name = module_info
        module = importlib.import_module(module_name)
        return getattr(module, command_name)

@cli.group(cls=LazyPipelineGroup)
@click.pass_context
def setup(ctx):
    '''Setup commands for the Galaxy Classification project.'''
    from src.utils.bootstrap import load_context
    
    ctx.obj = load_context(service="cli")
    