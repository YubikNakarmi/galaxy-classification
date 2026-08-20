from __future__ import annotations

import importlib

import click

from src.cli.style import UI

BANNER =r"""
✦   .        *       .    ✦        .
    *    .        ✦          *
  ✦        .    *       .        ✦

   ██████╗  █████╗ ██╗      █████╗ ██╗  ██╗
  ██╔════╝ ██╔══██╗██║     ██╔══██╗╚██╗██╔╝
  ██║  ███╗███████║██║     ███████║ ╚███╔╝
  ██║   ██║██╔══██║██║     ██╔══██║ ██╔██╗
  ╚██████╔╝██║  ██║███████╗██║  ██║██╔╝ ██╗
   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝

  ✦        .    *       .        ✦
    *    .        ✦          *
✦   .        *       .    ✦        .
"""

class LazyRootGroup(click.Group):# duplocated on sublevels instead of single import from /src to avaoid deps chaining
    """Load the top-level pipeline command only when it is requested."""

    command_modules = {
        "pipeline": ("ml.pipeline", "pipeline"),
        "setup": ("ml.setup", "setup"),
        "cache": ("ml.cache", "cache"),
        "status": ("ml.cli", "status"),
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

def get_version() -> str:
    from importlib.metadata import PackageNotFoundError,version
    try:
        return version("galaxy-classification")
    except PackageNotFoundError:
        return "dev"

@click.group(cls=LazyRootGroup,context_settings={
        "help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    help = "Command line interface for the Galaxy Classification project",
    epilog="Run galax COMMAND --help for more information on a command."
    )

@click.version_option(version=get_version(), prog_name="galaxy-classification")
@click.option("--verbose", "-v", count=True, help="Increase verbosity level (-v,-vv,-vvv).")
@click.pass_context
def cli(ctx,verbose):
    """Command line interface for the Galaxy Classification project."""
    ctx.obj = UI(verbose)  # to avaoid conflict with app contexxt used downstream
    ctx.meta["ui"] = ctx.obj #string typed extract with ui = cts["ui"]
    if ctx.invoked_subcommand is None:#for top-level command, show help if no subcommand is provided
        click.secho(BANNER, fg="cyan", bold=True)
        click.echo(ctx.get_help())




@cli.command()
@click.pass_context
def status(ctx):
    """Check the status for the project"""

    ui = ctx.obj or ctx.meta.get("ui") or UI(verbose=0)

    ui.rule("[bold green]Project Status[/bold green]", style="bold green",
            align="center",characters="─",width=80)

    ui.print("[bold cyan]Model Cache Summary: [/bold cyan]")
    

