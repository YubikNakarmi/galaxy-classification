'''
A module for managing model cache and metadata in MLflow. 
It provides functionality to load the best model from MLflow, with optional local caching, 
and to manage metadata about the models.

Classes:
    - ModelLoader: A class to load the best model from MLflow, with optional local caching.
    - Metadata: A base class for managing metadata about models.
    - ModelMetadata: A class for managing metadata specific to a model, including checksum generation.
'''

import hashlib
import json
import os
import mlflow
from mlflow.tracking import MlflowClient 
import torch
from typing import Optional,Any,Literal
from datetime import datetime, timezone
from loguru import logger
from pathlib import Path
from src.utils.schemas import CandidateMetadata,MetadataFile, VariantMetadata
from dataclasses import asdict
import hashlib

root = Path(__file__).parent.parent.parent

class Metadata:

    """A base class for managing metadata about models. It provides functionality to load and update metadata stored in a JSON file.

    Example:
        ```python
        metadata = Metadata(save_path="./cache/model_metadata.json")
        loaded_metadata = metadata._load()
        print(loaded_metadata.Default.latest_update)
        ```
    
    """
        
    def __init__(self,save_path:str|Path):
        """
        Args:
            save_path: The path to the JSON file where metadata will be stored.
            checksum: Whether to generate and store checksums for the metadata.

        """
    
        self.save_path = Path(save_path)


    def _load(self) -> MetadataFile: #load joson 
        """
        Load the metadata from the JSON file.

        Returns:
            An instance of MetadataFile containing the loaded metadata.
        Raises:
            FileNotFoundError: If the metadata file does not exist.
            ValueError: If the metadata file is not a valid JSON file.
        Example:
            ```python
            metadata = Metadata(save_path="./cache/model_metadata.json")
            loaded_metadata = metadata._load()
            print(loaded_metadata.Default.latest_update)
            ```
        """
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"File {self.save_path} does not exist.")
        with open(self.save_path, "r", encoding="utf-8") as file:
            try:
                data =  json.load(file)
            except json.JSONDecodeError:
                raise ValueError(f"File {self.save_path} is not a valid JSON file.")
            
        metadata = MetadataFile(
            Default=VariantMetadata(
                latest_update=data["Default"]["latest_update"],
                candidates=[
                    CandidateMetadata(**c)
                    for c in data["Default"]["candidates"]
                ],
            ),
            Hubble=VariantMetadata(
                latest_update=data["Hubble"]["latest_update"],
                candidates=[
                    CandidateMetadata(**c)
                    for c in data["Hubble"]["candidates"]
                ],
            ),
            Dirichlet=VariantMetadata(
                latest_update=data["Dirichlet"]["latest_update"],
                candidates=[
                    CandidateMetadata(**c)
                    for c in data["Dirichlet"]["candidates"]
                ],
            ),
        )

        return metadata


    def _write_json(self,file_path: str|Path, data: dict) -> None: #write to file
        """
        Write the given data to a JSON file.
        Args:
            file_path: The path to the JSON file where data will be written.
            data: A dictionary containing the data to be written to the JSON file.
        
        """

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=self._json_default)

    def _json_default(self,value: Any):# Convert datetime to ISO format, and handle other non-serializable types

        if isinstance(value, (datetime,)):
            return value.astimezone(timezone.utc).isoformat()
        if hasattr(value, "item"):
            return value.item()
        return str(value)
    


    def update_metadata(self,metadata:CandidateMetadata,
                        variant:Literal["Default","Hubble","Dirichlet"]="Default"):

        '''Update the metadata file with new candidate metadata for a specific variant.

        Args:
            metadata: An instance of CandidateMetadata containing the new candidate's information.
            variant: The variant for which to update the metadata. Must be one of "Default", "Hubble", or "Dirichlet".
        
        Example:
            ```python hl_lines = 8

            metadata = CandidateMetadata(
                rank=1,
                # ... other metadata fields
            )

            md = Metadata(save_path="./cache/model_metadata.json")
            md.update_metadata(metadata, variant="Default")
            ```
        
        '''

        try:
            #empty init
            if not self.save_path.exists() or self.save_path.stat().st_size == 0:
                logger.warning(f"Metadata file {self.save_path} missing or empty. Initializing with default structure.")
                file = MetadataFile()

            else:
                file = self._load()

            selected = file[variant]

            #sort by run id of existing
            existing = next((c for c in selected.candidates if c.run_id == metadata.run_id), None)

            if existing is not None: #if it exists remove 
                selected.candidates.remove(existing)
            selected.candidates.append(metadata)

            selected.candidates.sort(#sor the candidates after append
            key=lambda c: (c.f1_macro if c.f1_macro is not None else float("-inf")),
            reverse=True,)

            for i, c in enumerate(selected.candidates, start=1):
                c.rank = i

            selected.latest_update = datetime.now(timezone.utc).isoformat()

            self._write_json(self.save_path, asdict(file))
            logger.success(f"Metadata updated for variant '{variant}' at {self.save_path}")

        except Exception as e:
            logger.error(f"Error updating metadata: {e}")

    
    def __repr__(self):
        return f"Metadata(save_path={self.save_path!r})"


class ModelLoader:

    '''A class to load the best model from MLflow, with optional local caching.
    
    Example:
        ```python
        loader = ModelLoader(registered_model_name="my_model", local_cache=True, local_cache_dir="./cache", num_classes=5)
        model = loader.load_best_model()
        ```
 
    '''

    def __init__(self,registered_model_name:str,
                local_cache: bool = True,
                local_cache_dir: Optional[str| Path] = None,
                num_classes: int = 5,cache_metadata: bool = True, 
                metadata: Optional[Metadata]= None,
                metadata_save_path: Optional[str|Path] = "../../models/cached/metadata.json", 
                metric:str = "val_f1_macro",
                variant: Optional[Literal["Default","Hubble","Dirlichlet"]] = "Default",
                checksum_method: Literal["sha256","md5","sha1"] = "sha256",
                checksum: bool = True
                
                ):
        """
        Initialize the ModelLoader.

        Args:
            local_cache: Whether to use local caching.
            local_cache_dir: The directory for local caching.
            num_classes: The number of classes for the model.
            registered_model_name: The name of the registered model in MLflow.
            cache_metadata: Whether to cache metadata about the model. Recommended to be True for better validation and tracking of models.
            metadata: An instance of Metadata to manage model metadata. If None, a new Metadata instance will be created.
            metadata_save_path: The path to save the metadata.
            variant: The variant of the model for metadata. 
            checksum_method: The method to use for generating checksums. Options are "sha256", "md5", or "sha1".
            checksum: Whether to generate and store checksums for the model, also perform verification. Recommended to be True for data integrity verification.
        """
        

        self.client = MlflowClient()

        self.registered_model_name = registered_model_name  # Store the registered model name for later use
        self.local_cache = local_cache
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self.num_classes = num_classes
        self.cache_metadata = cache_metadata
        self.metadata_save_path = Path(metadata_save_path) if metadata_save_path else None
        self.best_run_id = None  # Store the best run ID for later use
        self.variant = variant  # Store the variant for metadata
        self.metric = metric  # Store the metric for sorting runs
        self.checksum_method = checksum_method  # Store the checksum method for metadata
        self.checksum = checksum

        
        if self.cache_metadata:
            if metadata is not None:
                self.metadata = metadata
            elif self.metadata_save_path is not None:
                self.metadata = Metadata(save_path=self.metadata_save_path)
            else:
                raise ValueError("metadata_save_path must be set if cache_metadata=True and no metadata instance is passed.")

        if self.cache_metadata:
            self.metadata = metadata if metadata else Metadata(save_path=self.metadata_save_path) 

        if self.local_cache and self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_metadata and not variant:
            logger.error("Variant must be provided if cache_metadata is True.")
            raise ValueError("Variant must be provided if cache_metadata is True.")

 

    def add_checksum(self,run_id:str)->str:

        if self.local_cache_dir is None:
            logger.error("local_cache_dir is not set; cannot compute checksum path.")
            raise RuntimeError("local_cache_dir is not set; cannot compute checksum path.")
        
        if self.checksum_method == "sha256":
            hash_func = hashlib.sha256()
        elif self.checksum_method == "md5":
            hash_func = hashlib.md5()
        else: 
            hash_func = hashlib.sha1()

        #hash the saved model binary
        with open(self.local_cache_dir / f"{run_id}.pt", "rb") as f:

            logger.info(f"Computing checksum for run '{run_id}'")
            for chunk in iter(lambda: f.read(81926), b""):#read in chunks to avoid memory issues with large model
                hash_func.update(chunk)

        return hash_func.hexdigest()
            
    
    def compare_checksum(self,calculated_checksum:str,current_checksum:str)->bool:
        return calculated_checksum == current_checksum

    
    def load_with_fallback(self,top_n:int=5) -> torch.nn.Module:
        '''Load the best model for the given experiment name, with a fallback to other models if best model fails.
        Args:
            top_n: The number of top runs to consider.
        Returns:
            The loaded PyTorch model.
        
        Raises:
            RuntimeError: If no model could be loaded from the top N runs.
        
        Example:
            ```python
            loader = ModelLoader(registered_model_name="my_model", local_cache=True, local_cache_dir="./cache", num_classes=5)
            model = loader.load_with_fallback(top_n=5)
            ```
        '''

      

        model_list = self._best_run(self.registered_model_name,top_n=top_n, metric=self.metric)
        if not model_list:
            raise RuntimeError(f"No runs found for '{self.registered_model_name}' to fall back on.")

        for rank,runs in enumerate(model_list,start=1):
            run_id = runs.info.run_id
            metric = runs.data.metrics.get(self.metric, None)

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
        ''' load the champion model from MLflow using the alias "champion". 
        If the alias is not found, it falls back to the latest run in the metadata for the specified variant.

        Returns:
            The loaded PyTorch model if successful, None otherwise.
        
        '''


        try:
            try:
                champion_version = self.client.get_model_version_by_alias(self.registered_model_name, "champion")
                champion_run_id = champion_version.run_id
            except Exception:
                logger.warning("Failed to load from MLflow alias; using local metadata to get run_id")
                met = self.metadata
                file = met._load()
                champion_run_id = file[self.variant].candidates[-1].run_id

            run = self.client.get_run(champion_run_id)
            f1_macro = run.data.metrics.get(self.metric, None)

            logger.success(f"Champion run ID: {champion_run_id}")

            if self.cache_metadata and self.local_cache_dir:
                met = CandidateMetadata(
                    rank=1,
                    run_id=champion_run_id,
                    model_name=self.registered_model_name,
                    experiment_name=self.registered_model_name,
                    f1_macro=f1_macro,
                    val_f1_macro=None, # Set to None or fetch from run data if available
                    num_classes=self.num_classes,
                    cache_file=str(self._cache_path(champion_run_id)),
                    cached_at=datetime.now(timezone.utc).isoformat(),
                    save_path=str(self.local_cache_dir),
                )
                self.metadata.update_metadata(met, variant=self.variant)

            return self.check_cache(champion_run_id)

        except Exception as e:
            logger.exception(f"Could not retrieve champion model for '{self.registered_model_name}': {e}")
            return None

  
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
        
        try:
            mlflow_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

              

            # extract state_dict from ExportedProgram and load into fresh model
            model = self._init_model()
            missing, unexpected = model.load_state_dict(mlflow_model.state_dict(), strict=False)

            if missing or unexpected:
                logger.warning(f"Missing: {missing}, Unexpected: {unexpected}")

            if self.local_cache and self.local_cache_dir:
                torch.save(model.state_dict(), cache_path)
                logger.info(f"Cached state_dict to {cache_path}")

                if self.checksum:
                    added_checksum = self.add_checksum(run_id)
                    self._save_checksum(run_id, added_checksum)#save checksum to save dir 
                    logger.info(f"Checksum ({self.checksum_method}) for run '{run_id}': {added_checksum}")

        except Exception as e:
            logger.exception(f"Failed to load model from MLflow for run '{run_id}': {e}")
            #azureblobfallback(run_id)
            raise e

        return model  


    def _save_checksum(self, run_id: str, checksum: str) -> None:

            
            if not self.metadata_save_path:
                logger.error("Metadata save path is not set; cannot save checksum.")
                raise
    
            checksum_path = self._checksum_path(run_id)
            with open(checksum_path, "w") as f:
                f.write(checksum)
            logger.success(f"Checksum saved to {checksum_path}")
   
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


    def _checksum_path(self, run_id: str) -> Path:
    
            '''
            Helper for checksum path. That depends on _cache_path and the checksum method. 
            It returns the path to the checksum file for a given run_id.
            
            Args:
                run_id: The MLflow run ID for which to compute the checksum path.
            Example:
                ```python
                >>> modelloader._checksum_path("12345")
                PosixPath('path/to/metadata/12345.sha256')
            '''
            if not self.metadata_save_path:
                logger.error("Metadata save path is not set; cannot compute checksum path.")
                raise
    
            return self._cache_path(run_id).with_suffix(f".{self.checksum_method}")

    def _init_model(self) -> torch.nn.Module:
            
            from src.models.model import GalaxyClassifier_ResNet
            return GalaxyClassifier_ResNet(N=2, k=2, num_classes=self.num_classes)# for proper mapping
    
    def _cache_path(self, run_id: str) -> Path:

        '''Compute the cache path for a given run_id.

        '''
        if self.local_cache_dir is None:
            raise RuntimeError("local_cache_dir is not set; cannot compute cache path.")

        return self.local_cache_dir / f"{run_id}.pt" 
    

    def __repr__(self):
        return f"ModelLoader(registered_model_name={self.registered_model_name!r}, \
        local_cache={self.local_cache}, local_cache_dir={self.local_cache_dir}, \
        num_classes={self.num_classes}, cache_metadata={self.cache_metadata}, \
        metadata_save_path={self.metadata_save_path}, \
        variant={self.variant})"

class LruCache:
    pass


