import mlflow
import bentoml
from mlflow.tracking import MlflowClient
from src.utils.yaml_utils import load_yaml
from pathlib import Path

root = Path(__file__).parent.parent


file_config = root / 'configs' / 'ml.yaml'
ml_yaml = load_yaml(file_config)
dev = ml_yaml["env"]["device"]
experiment_root = ml_yaml["variants"][ml_yaml["active_variant"]]["experiment"]["root"]


def best_modes_default():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("galaxy-classification-default-experiments")
    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.val_f1_macro DESC"], max_results=1)
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    return model



def promote_to_bentoml():


    