from __future__ import annotations

import click
from ml.cli import cli


class LazyPipelineGroup(click.Group):
    
    """Load pipeline commands only when they are invoked."""

    command_modules = {
        "modify": ("ml.cache.modify", "modify"),
        "import": ("ml.cache.modify", "import_model"),
        "show_local_cache":("ml.cache.summary","show_all")
        
    }

    def list_commands(self, ctx):
        return sorted(self.command_modules)

    def get_command(self, ctx, name):
        import importlib

        module_info = self.command_modules.get(name)
        if module_info is None:
            return None

        module_name, command_name = module_info
        module = importlib.import_module(module_name)
        return getattr(module, command_name)


@cli.group(cls=LazyPipelineGroup)
@click.pass_context
def cache(ctx):
    """Manage model cache.

    example:\n
        galax cache import --checkpoint_path <path> --variant <variant_name> --num_classes <num_classes> [--run_id <run_id>] [--file_dir <file_dir>]
    """

    from src.utils.bootstrap import load_context

    ctx.obj = load_context(service="ml",std_out=False)  # Load ML context for pipeline commands
