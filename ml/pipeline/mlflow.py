import click 
from typing import Optional

from ml.pipeline import pipeline
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

@pipeline.command(name="load-metrics")
@click.option("--run_id","-r",required=True,help="Run ID of the training run to load metrics for.")
@click.option("--save_path","-s",default=None,help="Directory or CSV path for the combined metrics file.")
@click.option("--metric_names","-m",multiple=True, required=False, default=None, help="List of metric names to load.")
@click.option("--exclude_system","-e",is_flag=True,default = True,help="Exclude system metrics.")
@click.pass_context
def load_metrics(ctx,run_id: str,
                 metric_names: Optional[list[str]]=None, 
                 save_path: Optional[str] = None, 
                 exclude_system: bool = True):
    
    from src.cli.style import UI
    from src.utils.bootstrap import load_context
    from src.training.metrics import load_mlflow_metrics
    
    app_ctx = ctx.obj or load_context(service="ml") #retrive from top level of click cmd in __init__py
    cfg = app_ctx.cfg
    
    ui = ctx.find_object(UI) or UI(verbose=0)
    
    ui.info(f"Loading metrics for run_id: {run_id}...")
    try:
        metrics = load_mlflow_metrics(
            run_id,
            list(metric_names) or None,
            save_path=save_path,
            ui=ui,
            exclude_system=exclude_system,
        )

    except Exception as e:
        ui.error(f"Failed to load metrics for run_id: {run_id}. Error: {str(e)}",show_execption=True)
        raise

def upload_missing_model():
    pass