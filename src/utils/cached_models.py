import hashlib
import json
import os
import mlflow
from mlflow.tracking import MlflowClient 
import torch
from typing import Optional,Any
from datetime import datetime, timezone


from loguru import logger

from pathlib import Path

root = Path(__file__).parent.parent.parent

class ModelLoader:

    '''A class to load the best model from MLflow, with optional local caching.
    
    Example:
        ```python
        loader = ModelLoader(registered_model_name="my_model", local_cache=True, local_cache_dir="./cache", num_classes=5)
        model = loader.load_best_model()
        ```
 
    '''

    def __init__(self,registered_model_name:str,local_cache: bool = True,
                 local_cache_dir: Optional[str| Path] = None,
                 num_classes: int = 5):
        """
        Initialize the ModelLoader.

        Args:
            local_cache: Whether to use local caching.
            local_cache_dir: The directory for local caching.
            num_classes: The number of classes for the model.
            registered_model_name: The name of the registered model in MLflow.
        """

        self.client = MlflowClient()

        self.registered_model_name = registered_model_name  # Store the registered model name for later use

        self.local_cache = local_cache
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self.num_classes = num_classes
        self.best_run_id = None  # Store the best run ID for later use

        if self.local_cache and self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    

    def load_with_fallback(self,top_n:int=5, metric: str = "val_f1_macro") -> torch.nn.Module:
        '''Load the best model for the given experiment name, with a fallback to other models if best model fails.
        Args:
            top_n: The number of top runs to consider.
            metric: The metric to sort runs by.
        Returns:
            The loaded PyTorch model.
        
        Raises:
            RuntimeError: If no model could be loaded from the top N runs.
        
        Example:
            ```python
            loader = ModelLoader(registered_model_name="my_model", local_cache=True, local_cache_dir="./cache", num_classes=5)
            model = loader.load_with_fallback(top_n=5, metric="val_f1_macro")
            ```
        '''

        model_list = self._best_run(self.registered_model_name,top_n=top_n, metric=metric)

        for rank,runs in enumerate(model_list,start=1):
            run_id = runs.info.run_id
            metric = runs.data.metrics.get(metric, None)

            logger.info(f"Attempting to load model from run '{run_id}' with {metric}: {metric} (Rank: {rank})")

            try: 
                model = self.check_cache(run_id)
                logger.success(f"Successfully loaded model from run '{run_id}' with {metric}: {metric} (Rank: {rank})")

                if model is None:
                    logger.exception(f"Model loading failed for run '{run_id}' with {metric}: {metric} (Rank: {rank})")

                if rank > 1:
                    logger.warning(f"Loaded model from run '{run_id}' with {metric}: {metric} (Rank: {rank}), which is not the best model.")

                return model
            
            except Exception as e:
                logger.error(f"Couldn't load the model from run_id:{run_id} continuing search")
                continue

        raise RuntimeError(f"Failed to load any model for experiment '{self.registered_model_name}' after checking top {top_n} runs.")
   
        

    def load_champion_model(self)->Optional[torch.nn.Module]:  
        '''Retrieve the champion model's run ID from MLflow.
        
        Returns:
            The run ID of the champion model if it exists, otherwise None.
        Raises:
            Exception: If there is an error retrieving the champion model.
        
        '''

        try:
            champion_version = self.client.get_model_version_by_alias(self.registered_model_name, "champion")
            champion_run_id = champion_version.run_id
            logger.success(f"Champion model version: {champion_version.version} | Run ID: {champion_run_id}")
            return self.check_cache(champion_run_id)
        
        except Exception as e:
            logger.exception(f"Could not retrieve champion model for '{self.registered_model_name}': {e}")
            return None


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

   
    def _best_run(self,model_name:str,top_n:int=5, metric: str = "val_f1_macro"):
            '''Retrieve the best run based on the highest specified metric.

            Args:
                model_name: The name of the model in Mlflow Registry.
                top_n: The number of top runs to consider.
                metric: The metric to sort runs by.

            Returns:
                A list of the top N runs based on the specified metric.

            '''
    
            experiment_name = self.client.get_experiment_by_name(model_name) 
    
            if experiment_name is None:
                logger.error(f"No experiment found with name '{model_name}'")
                return None
    
            runs = self.client.search_runs(
                experiment_ids=[experiment_name.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=[f"metrics.{metric} DESC"],
                max_results=top_n,
                )
    
            if not runs: 
                logger.error(f"No finished runs found for experiment '{model_name}'")
                return None
            return runs



class LruCache:
    pass

class Metadata:
    
    def __init__(self,save_path: Optional[str|Path] = None):
    
        self.save_path = Path(save_path) if save_path else None

    def _log_formats(self):

        top_json_format = {
            "Default": {
                "candidates":[],
                "latest_update": ""

                },
            "Hubble": {
                "candidates":[],
                "latest_update": ""

                },

            "Dirichlet": {
                "candidates":[],
                "latest_update": ""
            }
        }

        candidates_json_format =  {
            "rank": None,
            "run_id": "",
            "model_name": "",
            "experiment_name": "",
            "val_f1_macro": None,
            "num_classes": None,
            "cache_file": "",
            "cached_at": "",
            "save_path": ""
        }

        return top_json_format, candidates_json_format

    def _load_json(self, file_path: str) -> dict: #load joson 
        """
        Load a JSON file.
        """
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    def _write_json(self,file_path: str, data: dict) -> None: #write to file

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=self._json_default)

    def _json_default(self,value: Any):# Convert datetime to ISO format, and handle other non-serializable types

        if isinstance(value, (datetime,)):
            return value.astimezone(timezone.utc).isoformat()
        if hasattr(value, "item"):
            return value.item()
        return str(value)


    def update_metadata():
        pass



class ModelMetadata(Metadata):


    def __init__(self, run_id: str, model_name: str, metric: dict, save_path: Optional[str|Path] = None):

        super().__init__(save_path)
        self.model_type = "PyTorch"
        self.run_id = run_id
        self.model_name = model_name

    def checksum(self)-> str:
        '''Generate a checksum for the model metadata.'''
        data = f"{self.run_id}{self.model_name}{self.metric}"
        return hashlib.md5(data.encode()).hexdigest()

    def __str__(self):
        return f"ModelMetadata(run_id={self.run_id}, model_name={self.model_name}, metric={self.metric})"

   