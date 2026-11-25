""" for mkdocs-click docs generationg avoiding lazy loading of modules"""

from ml.cli import cli

ctx=None
for name in cli.list_commands(ctx):
    cli.add_command(cli.get_command(ctx, name))