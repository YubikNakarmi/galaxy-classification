from __future__ import annotations

import importlib

import click
from ml.cli import cli


class LazyPipelineGroup(click.Group):
    """Load pipeline commands only when they are invoked."""

    command_modules = {
        "train": ("ml.pipeline.train", "train"),
        "evaluate": ("ml.pipeline.eval", "evaluate"),
        "preprocess": ("ml.pipeline.preprocess", "preprocess"),
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
def pipeline(ctx):
    """ML pipeline commands.

    example:\n
        galax pipeline preprocess --run_preprocessing True \
        --preprocess_type classic
    """

    from src.utils.cli_bootstrap import load_context

    ctx.obj = load_context(service="ml")
