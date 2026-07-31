from pathlib import Path

import mlflow
import torch
from src.utils.yaml_utils import load_yaml
from src.config import load_and_validate_config
from src.utils.logging_config import setup_ml_logging
from src.utils.cached_models import ModelLoader
from src.dataset.dataset import GalaxyDataset
from src.training.metrics import make_metrics, val_transforms
from src.training.artifacts import promote_if_better

import pandas as pd
import datetime
import os
import click
from mlflow.tracking import MlflowClient



from loguru import logger
root = Path(__file__).resolve().parent.parent

file_config = root / 'configs' / 'ml.yaml'
setup_ml_logging(load_yaml(file_config))
ml_yaml = load_yaml(file_config)

cfg = load_and_validate_config(file_config)
active_var = cfg.active_variant
dev = cfg.env.device
experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


cache_dir = cfg.storage.local.checkpoints.model_cache_dir

def eval(metrics:dict,model:torch.nn.Module):
    logger.info(f"Loading model from cache dir: {cache_dir}")

    

    labels = pd.read_csv(cfg.data.eval_path)
    data_loader= GalaxyDataset(img_dir=cfg.data.img_dir, labels_df=labels, transform=val_transforms)

    with mlflow.start_run(run_name=f"eval_{experiment_name}"):

        model.eval()
        for img,labels in data_loader:
            with torch.no_grad():
                outputs = model(img)
                preds = torch.softmax(outputs, dim=1)

                for m in metrics.values():#update at each batch
                    m.update(preds, labels)
        results = {
        "test_acc":               metrics["accuracy"].compute().item(),
        "test_f1_micro":          metrics["f1_micro"].compute().item(),
        "test_f1_macro":          metrics["f1_macro"].compute().item(),
        "test_precision_macro":   metrics["precision_macro"].compute().item(),
        "test_recall_macro":      metrics["recall_macro"].compute().item(),
        "test_roc_auc":           metrics["roc_auc"].compute().item(),
    }

        for m in metrics.values():
            m.reset()

        return results


    

@click.command()
@click.option('--run_id', required=True, help='MLflow run ID to evaluate. If not provided, the best model will be used.')
@click.option('--variant', default=active_var, help='Variant of the experiment to evaluate.',type=click.Choice(["Default", "Hubble", "Dirichlet"]))
@click.option('--model_version', default=None, help='Version of the model to evaluate.')
@click.option("--min-improvement", type=float, default=0.005, help="Minimum metric improvement required to promote")
@click.option("--metric-name", type=str, default="test_f1_macro", help="Which computed metric drives the promotion decision")
@click.option("--alias", type=str,default="champion", help="Alias for the registered model")

def main(run_id, variant, model_version, min_improvement, metric_name, alias):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


    variant_cfg = cfg.variants[variant]
    experiment_root = variant_cfg.experiment.root



    client = MlflowClient()

    if run_id is None:

        raise ValueError("Run ID must be provided to evaluate a specific model.")

    model_name = None

    if model_version is None:
        # get the model version associated with the run_id
        ver = client.search_model_versions(f"run_id='{run_id}'")
        if not ver:
            logger.error("No registered version for run_id {} under variant {}", run_id, variant)
            return
        model_version = ver[0].version
        model_name = ver[0].name

    else:
        model_version = str(model_version)
        model_name = client.get_model_version(name=variant_cfg.mlflow.model_name, version=model_version).name


    model_uri = f"models:/{model_name}/{model_version}"   
    model = mlflow.pytorch.load_model(model_uri)

    # find which variant config this registered_model_name belongs to,
    matching_variant = next(
        (v for v in cfg.variants.values() if v.mlflow.model_name == model_name),
        None,
    )
    if matching_variant is None:
        logger.error("Could not match registered model '{}' to any variant in ml.yaml", model_name)
        return

    experiment_name = f"{matching_variant.experiment.root}_eval"
 

    mlflow.set_experiment(experiment_name)

    run_name = f"eval_{model_name}_v{model_version}_" + \
               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Evaluating model {model_name} version {model_version} from run {run_id}")
       
        metrics   = make_metrics(cfg.variant.num_classes, dev)
        results = eval(metrics, model)

        results = eval(
            metrics=metrics,
            model=model,
        )

        mlflow.log_metrics(results)
        logger.info("Eval results | {}", results)

        candidate_metric = results.get(metric_name)
        if candidate_metric is None:
            logger.error("Metric '{}' not present in eval results; cannot promote.", metric_name)
            return


        #promote the model if it meets the criteria
        promote_if_better(
            registered_model_name=model_name,
            candidate_version=model_version,
            candidate_metric=candidate_metric,
            metric_name=metric_name,
            alias=alias,
            min_improvement=min_improvement,
        )

if __name__ == "__main__":
    main()