from pathlib import Path

import mlflow
import torch
from src.utils.yaml_utils import load_yaml
from src.config import load_and_validate_config
from src.utils.logging_config import setup_ml_logging
from src.utils.cached_models import ModelLoader
from src.dataset.dataset import GalaxyDataset
from src.training.metrics import make_metrics, val_transforms
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


    

    # Add evaluation logic here
@click.command()
@click.option('--run_id', required=True, help='MLflow run ID to evaluate. If not provided, the best model will be used.')
@click.option('--variant', default=active_var, help='Variant of the experiment to evaluate.',type=click.Choice(["Default", "Hubble", "Dirichlet"]))
@click.option('--model_version', default=None, help='Version of the model to evaluate.')
@clicl.optoin("")

def main(run_id, variant, model_version):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


    variant_cfg = cfg.variants[variant]
    experiment_root = variant_cfg.experiment.root



    client = MlflowClient()

    if run_id is None:

        raise ValueError("Run ID must be provided to evaluate a specific model.")

    model_name = None

    if model_version is None:
        # get the latest version of the model 
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
 

    mlflow.set_experiment(experiment_root + "_eval")
    run_name = f"eval_{model_name}_v{model_version}_" + \
               datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Evaluating model {model_name} version {model_version} from run {run_id}")
       
        metrics   = make_metrics(cfg.variant.num_classes, dev)
        eval(metrics, model)

if __name__ == "__main__":
    main()