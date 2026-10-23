
import datetime
from pathlib import Path

import click
from src.cli.style import UI


from typing import TYPE_CHECKING
#type checking without importing on runtime
if TYPE_CHECKING:
    from src.config import RootConfig


from ml.pipeline import pipeline
root = Path(__file__).resolve().parent.parent

#no typing to lower the latnency of click cli command and avoid loading config multiple times in each command
def eval(metrics:dict,model,cfg,ui:UI)-> dict:

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import pandas as pd
    import torch
    from loguru import logger
    from src.dataset.dataset import GalaxyDataset
    from src.training.metrics import val_transforms
    from src.models.model import GalaxyClassifier_ResNet

    #Loading config context with better way than usign gloals which conflicts in cli
    cache_dir = cfg.storage.local.checkpoints.model_cache_dir
    experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) 
    logger.info(f"Starting evaluation for experiment: {experiment_name}")
    logger.info(f"Loading model from cache dir: {cache_dir}")
    num_class = cfg.variants[cfg.active_variant].num_classes
 
    
    labels = pd.read_csv(cfg.data.eval_path)

    data_loader = DataLoader(
        GalaxyDataset(img_dir=cfg.data.img_dir, labels_df=labels, transform=val_transforms),
        batch_size=32,  
        shuffle=False,
    )
    try:
        model.eval()
    except Exception as e:
        logger.error(f"Error occurred while perfoming evaluation: {e}")
        #
        model_skeleton = GalaxyClassifier_ResNet(N=2, k=2, num_classes=num_class)

        #map model
        missing, unexpected = model_skeleton.load_state_dict(model.state_dict(), strict=False)
        model_skeleton.eval()
        model = model_skeleton 


        if missing or unexpected:

            logger.warning(f"Missing: {missing}, Unexpected: {unexpected}")
            logger.exception(f"Error loading model state dict: {e}")

        else:
            logger.info("Model state dict loaded successfully into skeleton model.")
            logger.info("No missing or unexpected keys found. Proceeding with evaluation.")
    
    with ui.eval.session(len(data_loader)) as session:
        for img,labels in data_loader:
            
            with torch.no_grad():
                outputs = model(img)
                preds = torch.softmax(outputs, dim=1)

                for m in metrics.values():#update at each batch
                    m.update(preds, labels)
                    
                session.update_batch(
                    acc=metrics["accuracy"].compute().item(),
                    f1_macro=metrics["f1_macro"].compute().item(),)
                
    results = {
            "acc":               metrics["accuracy"].compute().item(),
            "f1_micro":          metrics["f1_micro"].compute().item(),
            "f1_macro":          metrics["f1_macro"].compute().item(),
            "precision_macro":   metrics["precision_macro"].compute().item(),
            "recall_macro":      metrics["recall_macro"].compute().item(),
            "roc_auc":           metrics["roc_auc"].compute().item(),
            }

    for m in metrics.values():
        m.reset()

    return results


    
@pipeline.command()
@click.option('--run_id', required=True, help='MLflow run ID to evaluate. If not provided, the best model will be used.')

@click.option('--variant', default="Default", help='Variant of the experiment to evaluate.',type=click.Choice(["Default", "Hubble", "Dirichlet"]))
@click.option('--model_version', default=None, help='Version of the model to evaluate.')
@click.option("--min-improvement", type=float, default=0.005, help="Minimum metric improvement required to promote")
@click.option("--metric_name", type=str, default="f1_macro", help="Which computed metric drives the promotion decision")
@click.option("--model_source", type=click.Choice(["mlflow", "cache"]), default="mlflow", help="Source of the model to evaluate (mlflow or cache)")
@click.option("--alias", type=str,default="Champion", help="Alias for the registered model")
@click.pass_context

def evaluate(ctx, run_id, variant, model_version, 
             min_improvement, metric_name, alias,model_source):
    #need to add model source for local eval and upload to registry
    '''
    
    
    '''
    from src.utils.bootstrap import load_context
    from src.cli.status import ConfigUI
    
    
    ui = ctx.find_object(UI) or UI(verbose=0)

    with ui.status("Loading configuration...") as status:

        import mlflow
        from loguru import logger
        from mlflow.tracking import MlflowClient

        from src.training.artifacts import promote_if_better
        from src.training.metrics import make_metrics
        from src.utils.bootstrap import load_context
        
        app_ctx = ctx.obj or load_context(service="ml") #retrive from top level of click cmd in __init__py
        cfg = app_ctx.cfg

        MLFLOW_TRACKING_URI = app_ctx.mlflow_tracking_uri


        status.update("Validating configuration and MLflow connection...")

        if Path(cfg.data.eval_path).exists() is False:
            logger.error("Evaluation data not found at {}. Please run preprocessing first.", cfg.data.eval_path)
            return

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Connection Test")
            with mlflow.start_run(run_name="setup_validation"):
                mlflow.log_param("validation", "success")
            logger.info(f"MLflow connection OK ({MLFLOW_TRACKING_URI})")

        except Exception as e:
            logger.error(f"MLflow not reachable: {e}")
            return
        
        active_var = cfg.active_variant
        dev = cfg.env.device
        experiment_name = cfg.active_variant + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))  


        logger.info(f"Starting evaluation for variant '{variant}' with run_id '{run_id}' and model_version '{model_version}'")
        ui.debug(f"Active variant: {active_var}, Device: {dev}, Experiment name: {experiment_name}")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



        logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        ui.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        variant_cfg = cfg.variants[variant]



        client = MlflowClient()

        if run_id is None:

            raise ValueError("Run ID must be provided to evaluate a specific model.")

        model_name = None

        if model_version is None:
            # get the model version associated with the run_id
            logger.info(f"Fetching model version for run_id '{run_id}' under variant '{variant}'")
            ui.info(f"Fetching model version for run_id '{run_id}' under variant '{variant}'")

            ver = client.search_model_versions(f"run_id='{run_id}'")
            if not ver:
                logger.error("No registered version for run_id {} under variant {}", run_id, variant)
                ui.error("No registered version for run_id {} under variant {}", run_id, variant)
                return
            model_version = ver[0].version
            model_name = ver[0].name

        else:
            logger.info(f"Using provided model_version '{model_version}' for variant '{variant}'")  
            ui.info
            model_version = str(model_version)
            model_name = client.get_model_version(name=variant_cfg.mlflow.model_name, version=model_version).name


        model_uri = f"models:/{model_name}/{model_version}"   

        logger.info(f"Loading model from MLflow URI: {model_uri}")
        ui.info(f"Loading model from MLflow URI: {model_uri}")

        model = mlflow.pytorch.load_model(model_uri)
        logger.info(f"Model loaded successfully from {model_uri}")  
        ui.info(f"Model loaded successfully from {model_uri}")

        # find which variant config this registered_model_name belongs to,
        matching_variant = next(
            (v for v in cfg.variants.values() if v.mlflow.model_name == model_name),
            None,
        )
        if matching_variant is None:
            logger.error("Could not match registered model '{}' to any variant in ml.yaml", model_name)
            ui.error("Could not match registered model '{}' to any variant in ml.yaml", model_name)

            return

        experiment_name = f"{matching_variant.experiment.root}_eval"
        ui.trace(f"Set MLflow experiment to '{experiment_name}' for evaluation run.")   
    

        mlflow.set_experiment(experiment_name)
        ui.info(f"Set MLflow experiment to '{experiment_name}' for evaluation run.")

        run_name = f"eval_{model_name}_v{model_version}_" + \
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ui.info(f"Starting MLflow run '{run_name}' for evaluation of model '{model_name}' version '{model_version}'")

        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Evaluating model {model_name} version {model_version} from run {run_id}")
        
            metrics   = make_metrics(cfg.variant.num_classes, dev)
            results = eval(metrics, model,cfg=cfg,ui=ui)

        
            mlflow.log_metrics(results)
            logger.info("Eval results | {}", results)

            candidate_metric = results.get(metric_name)
            if candidate_metric is None:
                logger.error("Metric '{}' not present in eval results; cannot promote.", metric_name)
                return

            mlflow.set_tag("eval_metric", metric_name)
            mlflow.set_tag("evaluated", "True")
            mlflow.set_tag("evaluated_model_name", model_name)


            #promote the model if it meets the criteria
            promote_if_better(
                registered_model_name=model_name,
                candidate_version=model_version,
                candidate_metric=candidate_metric,
                metric_name=metric_name,
                alias=alias,
                min_improvement=min_improvement,
            )

            client.set_registered_model_tag(name=model_name, key="evaluated", value="True")

        mlflow.end_run()

