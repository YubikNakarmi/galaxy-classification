
from __future__ import annotations

import datetime
from pathlib import Path
import click
from src.cli.style import UI
from typing import TYPE_CHECKING, Optional, Literal
from collections.abc import Callable

#type checking without importing on runtime
if TYPE_CHECKING:
    from src.config import RootConfig
    from torch.nn import Module
    from torchmetrics import Metric


from ml.pipeline import pipeline
root = Path(__file__).resolve().parent.parent



#no typing to lower the latnency of click cli command and avoid loading config multiple times in each command
def run_eval(
    model: Module,
    metrics: dict[str, Metric],
    cfg: RootConfig,
    dev: str,
    ui: UI,
    batch_size: int = 32,) -> dict[str, float]:
    """ Minimalist eval setup"""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.dataset.dataset import GalaxyDataset
    from src.training.metrics import val_transforms

    labels_df = pd.read_csv(cfg.data.eval_path)
    loader = DataLoader(
        GalaxyDataset(img_dir=cfg.data.img_dir, labels_df=labels_df, transform=val_transforms),
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(dev).eval()

    try:
        with ui.eval.session(len(loader)) as session:
            with torch.inference_mode():
                for img, target in loader:
                    img, target = img.to(dev), target.to(dev)
                    preds = torch.softmax(model(img), dim=1)

                    for m in metrics.values():
                        m.update(preds, target)

                    #cheap metrics for display
                    session.update_batch(
                        acc=metrics["accuracy"].compute().item(),
                        f1_macro=metrics["f1_macro"].compute().item(),
                    )

        results = {
            "acc": metrics["accuracy"].compute().item(),
            "f1_micro": metrics["f1_micro"].compute().item(),
            "f1_macro": metrics["f1_macro"].compute().item(),
            "precision_macro": metrics["precision_macro"].compute().item(),
            "recall_macro": metrics["recall_macro"].compute().item(),
            "roc_auc": metrics["roc_auc"].compute().item(),
        }
    finally:
        for m in metrics.values():
            m.reset()

    return results

    
@pipeline.command()
@click.option('--run_id', required=True, 
              help='MLflow run ID to evaluate. If not provided, the best model will be used.')

@click.option('--variant', default="Default", 
              help='Variant of the experiment to evaluate.',type=click.Choice(["Default", "Hubble", "Dirichlet"]))
@click.option('--model_version', default=None, 
              help='Version of the model to evaluate.')
@click.option("--min-improvement", type=float, default=0.005, 
              help="Minimum metric improvement required to promote")
@click.option("--metric_name", type=str, default="f1_macro", 
              help="Which computed metric drives the promotion decision")
@click.option("--model_source", type=click.Choice(["mlflow", "local"]), default="mlflow", 
              help="Source of the model to evaluate (mlflow or local)")
@click.option("model_path","-m", type = click.Path(exists=True),
              help="Path to the model file to evaluate",default=None)
@click.option("--alias", type=str,default="Champion", help="Alias for the registered model")

@click.option("--register", is_flag=True, default=True,
              help="Register the model to MLflow if it is missing")
@click.pass_context

def evaluate(ctx, 
             run_id:click.Context, 
             variant:Literal["Default", "Hubble", "Dirichlet"],
             model_version:Optional[str], 
             min_improvement:float, 
             metric_name:str, 
             alias:str,
             model_source:Literal["mlflow", "local"],
             model_path:Optional[str],
             register:bool):
    '''
    Evaluate a model on a given dataset.
    
    '''
    
    #args check for mismatch
    if model_source is None:
        model_source = "local" if model_path else "mlflow"
    if model_source == "local" and not model_path:
        raise click.UsageError("--model_source local requires -m <path>.")
    if model_source == "mlflow" and model_path:
        raise click.UsageError("-m <path> cannot be combined with --model_source mlflow.")
    
    from src.utils.bootstrap import load_context
    
    app_ctx = ctx.obj or load_context(service="ml") #retrive from top level of click cmd in __init__py
    cfg = app_ctx.cfg
    ui = ctx.find_object(UI) or UI(verbose=0)  #init ui for cli
    
    MLFLOW_TRACKING_URI = app_ctx.mlflow_tracking_uri


    with ui.status("Loading configuration...") as status:

        import mlflow
        from mlflow.tracking import MlflowClient

        from src.training.artifacts import (promote_if_better,
                                            should_promote,
                                            get_champion,load_local_model,
                                            upload_model,_timestamp
                                            )
        from src.training.metrics import make_metrics
        from src.training.validation import validate_setup, find_model
        

        status.update("Validating configuration and MLflow connection...")
        validate_setup(required_paths=[cfg.data.img_dir, cfg.data.eval_path], 
                       mlflow_tracking_uri=MLFLOW_TRACKING_URI)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        try:
            client.search_experiments(max_results=1)  # read-only connectivity check
        except Exception as e:
            raise click.ClickException(f"MLflow not reachable at {MLFLOW_TRACKING_URI}: {e}") from e
        
        dev = cfg.env.device
        variant_cfg = cfg.variants[variant]
        registry_name= variant_cfg.mlflow.model_name
        ui.info(f"MLflow: {MLFLOW_TRACKING_URI} | variant: {variant} | device: {dev}")
        
        model: Module
        model_name: str
        
        # Set only when the model isnt in the registry yet
        register_fn: Callable[[], str] | None = None
        
        status.update(f"Loading model from {model_source}...")
        if model_source == "mlflow":
            
            if model_version is not None:  #check form argument and directly get the model
                model_version = str(model_version)
                model_name = client.get_model_version(registry_name, model_version).name
                model_uri = f"models:/{model_name}/{model_version}"  #main link that loads model
                
            else:
                state, info = find_model(client, run_id)  #check if its registered if not provided in arg
                ui.info(f"Run '{run_id}' model state: {state}")

                if state == "registered":
                    model_name, model_version = info.name, str(info.version)
                    model_uri = f"models:/{model_name}/{model_version}"
                elif state == "logged":  #in the case if model is logged in run but not resigered
                    model_name = registry_name
                    model_uri = f"runs:/{run_id}/{str(info).rstrip('/')}"

                    def register_fn(uri: str = model_uri) -> str:  #define register instead of call
                        return str(mlflow.register_model(uri, registry_name).version)

                else:
                    raise click.ClickException(
                        f"No model found on run {run_id}. "
                        "Pass a checkpoint with -m <path> (and --register to upload it)."
                    )

            ui.info(f"Loading model from {model_uri}")
            
            status.update(f"Loading model from MLflow registry: {model_uri}")
            model = mlflow.pytorch.load_model(model_uri)

        else: #local
            from src.models.model import GalaxyClassifier_ResNet

            ui.info(f"Loading local model from {model_path}")
            
            status.update(f"loading local model from {model_path}")
            model = load_local_model(
                model_path, 
                lambda: GalaxyClassifier_ResNet(N=2, k=2, num_classes=variant_cfg.num_classes),
            )
            model_name = registry_name

            def register_fn(eval_id: str) -> str:
                return upload_model(model, registry_name, source_path=model_path,source_run_id = eval_id)

        matching_variant = next(
        (v for v in cfg.variants.values() if v.mlflow.model_name == model_name), None
        )
        
        if matching_variant is None:
            raise click.ClickException(
                f"Could not match registered model '{model_name}' to any variant in ml.yaml."
            )
        
        experiment_name = f"eval_{matching_variant.experiment.root}"
        mlflow.set_experiment(experiment_name)
        run_name = f"eval_{model_name}_v{model_version or 'unregistered'}_{_timestamp()}"
        ui.info(f"Experiment '{experiment_name}', run '{run_name}'")
        
        metrics = make_metrics(matching_variant.num_classes, dev)
        
        #eval run
    with mlflow.start_run(run_name=run_name) as run:
        eval_run_id: str = run.info.run_id
        ui.info(f"Evaluating {model_name} (source run {run_id})")

        results = run_eval(model, metrics, cfg, dev, ui)
        mlflow.log_metrics(results)
        ui.info(f"Eval results | {results}")

        candidate_metric = results.get(metric_name) #metric name by comparision
        if candidate_metric is None:
            raise click.ClickException(
                f"Metric '{metric_name}' not in eval results {sorted(results)}; cannot promote."
            )

        mlflow.set_tags(
            {
                "eval_metric": metric_name,
                "evaluated": "True",
                "evaluated_model_name": model_name,
                "source_run_id": run_id,
                "model_source": model_source,
                "eval_data": str(cfg.data.eval_path),
            })
        
    promoted = False
    reason = "not_registered"

    status.update(f"Checking if model '{model_name}' should be promoted to '{alias}'...")
    if model_version is not None:
        # Already in the registry
        ui.info(f"Model '{model_name}' is already registered.")
        promoted = promote_if_better(  #check if better and priomote
            registered_model_name=model_name,
            candidate_version=model_version,
            candidate_metric=candidate_metric,
            metric_name=metric_name,
            alias=alias,
            min_improvement=min_improvement,
        )
        reason = "promoted" if promoted else "not_better"
    else:
        # Not registered, and check if it should be prmoted
        ok, reason = should_promote(
            client, model_name, alias, metric_name, candidate_metric, min_improvement)
        
        if ok and register:
            assert register_fn is not None  #check if func is declared
            model_version = register_fn()
            client.set_model_version_tag(model_name, model_version, metric_name, str(candidate_metric))
            client.set_model_version_tag(model_name, model_version, "promoted_reason", reason)
            client.set_registered_model_alias(model_name, alias, model_version)
            promoted = True
            ui.success(f"Registered '{model_name}' v{model_version} and promoted to '{alias}' ({reason}).")
            
            
        elif ok:
            ui.info(f"Would promote ({reason}), but --register wasn't given. Nothing registered.")
            
        else:
            ui.info(f"Not promoting ({reason}). Nothing registered.")

    # Link the registry version and the eval run to each other.
    if model_version is not None:
        client.set_model_version_tag(model_name, model_version, "evaluated", "True")
        client.set_model_version_tag(model_name, model_version, "eval_run_id", eval_run_id)
        client.set_tag(eval_run_id, "evaluated_model_version", str(model_version))
        
    client.set_tag(eval_run_id, "promoted", str(promoted))
    client.set_tag(eval_run_id, "promotion_reason", reason)

    ui.info(f"Done. {metric_name}={candidate_metric:.4f} | promoted={promoted} ({reason})")
    
    return candidate_metric, promoted, reason  #for call elsewhere