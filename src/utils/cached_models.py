import mlflow
from mlflow.tracking import MlflowClient 
import torch
from typing import Optional
from loguru import logger

from pathlib import Path

root = Path(__file__).parent.parent.parent

class ModelLoader:

    '''A class to load the best model from MLflow, with optional local caching.
    
    Example:
        ```python
        loader = ModelLoader(experiment_name="my_experiment", local_cache=True, local_cache_dir="./cache", num_classes=5)
        model = loader.load_best_model()
        ```
 
    '''

    def __init__(self,experiment_name: str,local_cache: bool = True,
                 local_cache_dir: Optional[str| Path] = None,
                 num_classes: int = 5,registered_model_name: Optional[str] = None):
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
        self.num_classes = num_classes
        self.best_run_id = None  # Store the best run ID for later use
        self.registered_model_name = registered_model_name  # Store the registered model name for later use

        if self.local_cache and self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _best_run(self):
        '''Retrieve the best run based on the highest val_f1_macro metric.'''

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
        
        self.best_run_id = runs[0].info.run_id  # Store the best run ID

        return runs[0]

    def latest_registered(self):
        '''Retrieve the latest registered model for the given experiment.'''

        model = self.client.get_registered_model(self.registered_model_name)
        return model


    def get_champion_model(self):
        pass

    def register_champion_model




    def _init_model(self) -> torch.nn.Module:
        from src.models.model import GalaxyClassifier_ResNet
        return GalaxyClassifier_ResNet(N=2, k=2, num_classes=self.num_classes)# for proper mapping

    def _cache_path(self, run_id: str) -> Path:
        return self.local_cache_dir / f"{run_id}.pt" 
    
    def check_cache(self, run_id: str) -> torch.nn.Module:
        '''Check if the model for the given run_id is cached locally. If so, load it; otherwise, pull from MLflow.
        Args:
            run_id: The MLflow run ID for which to check the cache.
        Returns:
            The loaded PyTorch model.
        
        '''

        cache_path = self._cache_path(run_id)

        if self.local_cache and self.local_cache_dir and cache_path.exists():
            logger.info(f"Loading state_dict from cache: {cache_path}")
            state_dict = torch.load(cache_path, map_location="cpu", weights_only=True)
            model = self._init_model()
            model.load_state_dict(state_dict)
            return model

        logger.info(f"Cache miss for run '{run_id}', pulling from MLflow")
        mlflow_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

        # extract state_dict from ExportedProgram and load into fresh model
        model = self._init_model()
        missing, unexpected = model.load_state_dict(mlflow_model.state_dict(), strict=False)
        if missing or unexpected:
            logger.warning(f"Missing: {missing}, Unexpected: {unexpected}")

        if self.local_cache and self.local_cache_dir:
            torch.save(model.state_dict(), cache_path)
            logger.info(f"Cached state_dict to {cache_path}")

        return model  

    def load_best_model(self) -> torch.nn.Module:
        ''' Wraps around check_cache and best run to load the best model from mlflow or local cache
        
        Returns:
            The loaded PyTorch model corresponding to the best run.
        '''
        best_run = self._best_run()
        run_id = best_run.info.run_id
        logger.info(f"Best run: {run_id} | val_f1_macro: {best_run.data.metrics.get('val_f1_macro')}")
        model = self.check_cache(run_id)
        if model is None:
            logger.exception(f"Model not found for run ID '{run_id}'")

        return model
    
  