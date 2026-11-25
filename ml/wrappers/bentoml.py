import click
from ml.cli import cli
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


@cli.group()
@click.pass_context
def bentoml(ctx):
    """BentoML Wrapper"""
    from src.utils.bootstrap import load_context
    
    ctx.obj = load_context(service="ml",std_out=False)  # Load ML context for pipeline commands
    
    
@bentoml.command(name="containerize")
@click.pass_context
@click.option("--bentofile-path","-f", default=ROOT / "bentofile.yaml", type=str, required=True,
              help="Path to the BentoML configuration file (BentoFile)")
@click.option("--tag", "-t", default="latest", type=str, required=False, 
              help="Tag for the Docker image")

def containerize(ctx, bentofile_path:str, tag:str):
    """Containerize a BentoML service using the specified BentoFile and tag."""
    from src.cli.style import UI
    from src.utils.bootstrap import load_context
    
    app_ctx = ctx.obj or load_context(service="ml",std_out=False)
    cfg = app_ctx.cfg
    
    ui = ctx.find_object(UI) or UI(verbose=0)
    
    try:
        bentofile_path = Path(bentofile_path).resolve()
        if bentofile_path.is_dir():
            bentofile_path /= "bentofile.yaml"
        
        with ui.status(f"Building and containerizing BentoML service from {bentofile_path} with tag {tag} ..."):
            cont = subprocess.run(["bentoml","build","-f", str(bentofile_path),"--version", tag,"--containerize"],
                            capture_output=True, text=True)
            
        #hidden behind verbose level 2
        ui.debug(f"Command output: {cont.stdout}")

        if cont.returncode != 0:
            ui.error(f"Error during BentoML containerization: {cont.stderr}")
            raise RuntimeError(f"BentoML containerization failed with return code {cont.returncode}")
        
        ui.success(f"BentoML service containerized successfully with tag {tag}.")
    
    except Exception as e:
        ui.error(f"Error during BentoML containerization: {e}", show_execption=True)
        raise click.ClickException(f"Error during BentoML containerization: {e}")