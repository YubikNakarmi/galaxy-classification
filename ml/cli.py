from __future__ import annotations

import importlib

import click


class LazyRootGroup(click.Group):
    """Load top-level commands only when they are requested."""

    command_modules = {
        "pipeline": ("ml.pipeline", "pipeline"),
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


@click.group(cls=LazyRootGroup)
def cli():
    """Command line interface for the Galaxy Classification project."""
    pass
