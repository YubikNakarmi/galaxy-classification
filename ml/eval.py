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

def eval(metrics:dict):
    logger.info(f"Loading model from cache dir: {cache_dir}")

    loader = ModelLoader(experiment_name= cfg.variants[active_var].experiment.root,
                         local_cache_dir=cache_dir)
    
    model = loader.load_best_model()

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

def main():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.variants[active_var].experiment.root + "_eval")
    metrics   = make_metrics(cfg.variant.num_classes, dev)
    eval(metrics)

if __name__ == "__main__":
    main()