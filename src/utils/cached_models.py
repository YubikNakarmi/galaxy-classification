import mlflow
from mlflow.tracking import MlflowClient
import torch
from typing import Optional
from loguru import logger
from pathlib import Path

class ModelLoader:

    def __init__(self,experiment_name: str,local_cache: bool = True,local_cache_dir: Optional[str| Path] = None):
        """
        Initialize the ModelLoader.

        Args:
            experiment_name: The name of the MLflow experiment.
            local_cache: Whether to use local caching.
            local_cache_dir: The directory for local caching.
        """

        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self.local_cache = local_cache
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None

        if self.local_cache and self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _best_run(self):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found")

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.val_f1_macro DESC"],
            max_results=1,
        )

        if not runs:
            raise ValueError(f"No finished runs in experiment '{self.experiment_name}'")

        return runs[0]
    

    def _cache_path(self, run_id: str) -> Path:
        return self.local_cache_dir / f"model_{run_id}.pt"
    
    def check_cache(self, run_id: str) -> torch.nn.Module:
        ''' Checks local file dir if there is any present model for the best run_id or run_id provided'''
        cache_path = self._cache_path(run_id)

        if self.local_cache and self.local_cache_dir and cache_path.exists():
            logger.info(f"Loading model from local cache: {cache_path}")
            return torch.load(cache_path, weights_only=False)

        logger.info(f"Cache miss for run '{run_id}', pulling from MLflow")
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

        if self.local_cache and self.local_cache_dir:
            torch.save(model, cache_path)
            logger.info(f"Cached model to {cache_path}")

        return model

    def load_best_model(self) -> torch.nn.Module:
        ''' Wraps around check_cache and best run to load the best model from mlflow or local cache'''
        best_run = self._best_run()
        run_id = best_run.info.run_id
        logger.info(f"Best run: {run_id} | val_f1_macro: {best_run.data.metrics.get('val_f1_macro')}")
        model = self.check_cache(run_id)
        if model is None:
            logger.exception(f"Model not found for run ID '{run_id}'")

        return model
    
    def sync_model_context(self, model: torch.nn.Module, exported_model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """
        Synchronize the context of the exported model with the original model structure.
        
        Args:
            model: The original PyTorch model.
            exported_model: The exported PyTorch model (ExportedProgram).
        """
        state_dict = exported_model.state_dict()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing or unexpected : #remap properly to avoid missing keys
            logger.critical(f"Missing keys in state_dict: {missing}")
            logger.critical(f"Unexpected keys in state_dict: {unexpected}")
            return None
        

        return model.eval()  # Ensure the model is in evaluation mode after loading weights