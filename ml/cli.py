from __future__ import annotations

import importlib

import click


class LazyRootGroup(click.Group):
    """Load the top-level pipeline command only when it is requested."""

    command_modules = {
        "pipeline": ("ml.pipeline", "pipeline"),
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
