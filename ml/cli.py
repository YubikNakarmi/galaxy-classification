from __future__ import annotations

import importlib

import click

from src.cli.style import UI
from src.cli.utils import LazyPipelineGroup


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

command_modules = {
    "pipeline": ("ml.pipeline", "pipeline"),
    "setup": ("ml.setup", "setup"),
    "cache": ("ml.cache", "cache"),
    "status": ("ml.cli", "status"),
    "log":("ml.cli","log"),
    "bentoml":("ml.wrappers.bentoml","bentoml")
}

class LazyRootGroup(LazyPipelineGroup):
    """Load root commands only when they are invoked."""
    command_modules = command_modules
  
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
    
    ui = UI(verbose)

    ctx.obj = ui  # to avaoid conflict with app contexxt used downstream
    ctx.meta["ui"] = ctx.obj #string typed extract with ui = cts["ui"]
    if ctx.invoked_subcommand is None:#for top-level command, show help if no subcommand is provided
        click.secho(BANNER, fg="cyan", bold=True)
        click.echo(ctx.get_help())
            
   


@cli.command()
@click.pass_context
# @click.option("--section", "-s",
#                 multiple = True,
#                 type=click.Choice(["all","summary","env","azure"]), 
#                 default="ml",
#                 help="Section to show status for.",
#                 default = ("all",))
                
def status(ctx):
    """Check the status for the project"""

    ui = ctx.obj or ctx.meta.get("ui") or UI(verbose=0)

    from src.utils.bootstrap import load_context
    from src.cli.status import ConfigUI
    config_ui = ConfigUI(ui)
    ctx = load_context(service="ml",std_out=False)
    
    
    try:
        config_ui.show_global_config(ctx.cfg,mlflow_uri=ctx.mlflow_tracking_uri,
                                 banner=BANNER)
    except Exception as e:
        ui.error(f"Error while showing global configuration: {e}",show_execption=True)
        raise click.ClickException(f"Error while showing global configuration: {e}")
   

   

@cli.group()
@click.pass_context
def log(ctx):
    """Log management commands"""
    from src.utils.bootstrap import load_context
    
    ctx.obj = load_context(service="ml",std_out=False)  # Load ML context for pipeline commands
    

@log.command("show")
@click.pass_context
@click.option("--type","-t", default="ml",type=click.Choice(["ml","app"]),required=False)
@click.option("--level","-l", type=click.Choice(["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]))
@click.option("--lines","-n",default=50 ,type=int)
@click.option("--service","-s",default=None,help="filter by service")

def log_show(ctx,type,level,lines,service):
    """Show logs for the project"""

    from loguru import logger
    from src.utils.yaml_utils import load_yaml
    from pathlib import Path

    #log format in regex
    LOG_PATTERN = (
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"(?P<level>\w+)\s*\| "
    r"(?P<service>[^|]+?) \| "
    r"(?P<name>[^:]+):(?P<function>[^:]+):(?P<line>\d+) - "
    r"(?P<message>.*)")



    ROOT = Path(__file__).resolve().parent.parent 



    ui = ctx.find_object(UI) or UI(verbose=1)
    
    ml_yaml_path = ROOT / "configs" / "ml.yaml"
    core_yaml_path = ROOT / "configs" / "core.yaml"

    if type=="ml":
        ui.trace("using ml mode")
        core_cfg = load_yaml(core_yaml_path)
        log_path =  ROOT / core_cfg["logging"]["ml"]["file"]

        if not log_path.exists():
            ui.warning(f"No log file found at {log_path}")
            return

        records = list(logger.parse(str(log_path), LOG_PATTERN))

        if level:
            records = [r for r in records if r["level"].upper() == level.upper()]
        #filter services if needed
        if service:
            records = [r for r in records if r["service"].strip() == service]

        for r in records[-lines:]:
            lvl = r["level"].upper()
            loc = f"{r['name']}:{r['function']}:{r['line']}"
            text = f"{r['time']} [{lvl}] ({loc}) {r['message']}"

            if lvl == "ERROR":
                ui.console.print(f"[error]{text}[/error]")
            elif lvl == "WARNING":
                ui.console.print(f"[warning]{text}[/warning]")
            elif lvl == "SUCCESS":
                ui.console.print(f"[success]{text}[/success]")
            else:
                ui.console.print(f"[dim]{text}[/dim]")
                
    elif type=="app":
        ui.trace("using app mode")
        core_cfg = load_yaml(core_yaml_path)
        log_path =  ROOT / core_cfg["logging"]["app"]["file"]

        if not log_path.exists():
            ui.warning(f"No log file found at {log_path}")
            return

        records = list(logger.parse(str(log_path), LOG_PATTERN))

        if level:
            records = [r for r in records if r["level"].upper() == level.upper()]
        #filter services if needed
        if service:
            records = [r for r in records if r["service"].strip() == service]

        for r in records[-lines:]:
            lvl = r["level"].upper()
            loc = f"{r['name']}:{r['function']}:{r['line']}"
            text = f"{r['time']} [{lvl}] ({loc}) {r['message']}"

            if lvl == "ERROR":
                ui.console.print(f"[error]{text}[/error]")
            elif lvl == "WARNING":
                ui.console.print(f"[warning]{text}[/warning]")
            elif lvl == "SUCCESS":
                ui.console.print(f"[success]{text}[/success]")
            else:
                ui.console.print(f"[dim]{text}[/dim]")

        

        



