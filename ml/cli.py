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

    from rich.table import Table
    from rich import box

    from src.cli.style import UI

    variants = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    variants.add_column(style="bold")
    variants.add_column()
    variants.add_row("[bold yellow] Variants:[/bold yellow]", "")
    variants.add_row("  Default", f"[success]✓[/success] 5 candidates | champion: 9e2fb457... (val_f1_macro=0.217) | updated 2026-08-06")
    variants.add_row("  Hubble", "[warning]⚠[/warning] no candidates yet")
    variants.add_row("  Dirichlet", "[warning]⚠[/warning] no candidates yet")

    variants.add_row("\n[bold]Data:[/bold]", "")
    variants.add_row("  Images", "[success]✓[/success] 5,000 images | 5,000 downloaded | 5,000 verified")
    ui.console.print(variants)    

    variants.add_row("\n[bold]Environment:[/bold]", "")
    variants.add_row("  Python", "[error]✓[/error] Python 3.11.4 | venv: galaxy-classification-dev")

    varaints.add_row("")