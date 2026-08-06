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
from pathlib import Path
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
import json
from src.config import RootConfig

root = Path(__file__).parent.parent.parent

class Metadata:

    """A base class for managing metadata about models. It provides functionality to load and update metadata stored in a JSON file.

    Example:
        ```python
        metadata = Metadata(save_file_path="./cache/model_metadata.json")
        loaded_metadata = metadata._load()
        print(loaded_metadata.Default.latest_update)
        ```
    
    """
        
    def __init__(self,save_file_path:str|Path):
        """
        Args:
            save_file_path: The path to the JSON file where metadata will be stored.
            checksum: Whether to generate and store checksums for the metadata.

        """
    
        self.save_path = Path(save_file_path)
        if self.save_path.exists() and not self.save_path.is_file():
            raise FileNotFoundError(f"{self.save_path} is not a file")
        
        if not self.save_path.exists():
            os.makedirs(self.save_path.parent, exist_ok=True)
           

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
            metadata = Metadata(save_file_path="./cache/model_metadata.json")
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

    def __len__(self):
        """
        Return the total number of candidates across all variants in the metadata.
        Returns:
            The total number of candidates across all variants.
        Example:
            ```python
            metadata = Metadata(save_path="./cache/model_metadata.json")
            total_candidates = len(metadata)
            print(f"Total candidates: {total_candidates}")
            ```
        """
        file = self._load()
        return sum(len(getattr(file, variant).candidates) for variant in ["Default", "Hubble", "Dirichlet"])
    


    def update_metadata(self,
                        metadata:CandidateMetadata,metric:str,
                        variant:Literal["Default","Hubble","Dirichlet"]="Default",
                        top_n:int=5)->Any:

        '''Update the metadata file with new candidate metadata for a specific variant.

        Args:
            metadata: An instance of CandidateMetadata containing the new candidate's information.
            variant: The variant for which to update the metadata. Must be one of "Default", "Hubble", or "Dirichlet".
            metric: The metric to use for sorting the candidates.
        
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

            selected.candidates.sort(#sor the candidates after append by metrics
            key=lambda c: (c.metric.get(metric, 0) if c.metric.get(metric, 0) is not None else float("-inf")),
            reverse=True,)

            #drop candidates beyond top_n
            dropped = selected.candidates[top_n:]
            selected.candidates = selected.candidates[:top_n]

            for i, c in enumerate(selected.candidates, start=1):
                c.rank = i

            if dropped:
                logger.info(f"Pruned {len(dropped)} candidate(s) beyond top_n={top_n} for variant '{variant}': "
                       f"{[c.run_id for c in dropped]}")


            selected.latest_update = datetime.now(timezone.utc).isoformat()

            self._write_json(self.save_path, asdict(file))
            logger.success(f"Metadata updated for variant '{variant}' at {self.save_path}")

            return dropped #for file eviction

        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            raise

    
    def __repr__(self):
        return f"Metadata(save_path={self.save_path!r})"


class ModelLoader:

    '''A class to load the best model from MLflow, with optional local caching.
    
        Example: Minimal usage (no caching, no metadata)
        ```python
                loader = ModelLoader(
                    registered_model_name="galaxy-classification-default",
                    local_cache=False,
                    cache_metadata=False,
                )
                model = loader.load_with_fallback(top_n=5)
        ```

            Example: Local caching only (no metadata tracking)
        ```python
                loader = ModelLoader(
                    registered_model_name="galaxy-classification-default",
                    local_cache=True,
                    local_cache_dir="./models/cached",
                    num_classes=5,
                    cache_metadata=False,
                )
                model = loader.load_with_fallback(top_n=5)
        ```

            Example: Full setup — caching + metadata + checksum verification (recommended)
        ```python
                loader = ModelLoader(
                    registered_model_name="galaxy-classification-default",
                    local_cache=True,
                    local_cache_dir="./models/cached",
                    num_classes=5,
                    cache_metadata=True,
                    metadata_save_path="./models/cached/metadata.json",
                    variant="Default",
                    metric="val_f1_macro",
                    checksum=True,
                    checksum_method="sha256",
                )
                model = loader.load_with_fallback(top_n=5)
        ```

            Example: Loading the champion model directly (BentoML service startup)
        ```python
                loader = ModelLoader(
                    registered_model_name="galaxy-classification-default",
                    local_cache=True,
                    local_cache_dir="./models/cached",
                    num_classes=5,
                    variant="Default",
                )
                model = loader.load_champion_model()
                if model is None:
                    raise RuntimeError("No champion model available; check MLflow connectivity")
                model.eval()
        ```

            Example: Passing a pre-configured Metadata instance (shared across variants)
        ```python
                shared_metadata = Metadata(save_path="./models/cached/metadata.json")

                default_loader = ModelLoader(
                    registered_model_name="galaxy-classification-default",
                    local_cache_dir="./models/cached",
                    num_classes=5,
                    metadata=shared_metadata,
                    variant="Default",
                )
                hubble_loader = ModelLoader(
                    registered_model_name="galaxy-classification-hubble",
                    local_cache_dir="./models/cached",
                    num_classes=10,
                    metadata=shared_metadata,
                    variant="Hubble",
                )
        ```

            Example: Syncing metadata without loading a model (e.g. a periodic refresh job)
        ```python
                loader = ModelLoader(
                    registered_model_name="galaxy-classification-default",
                    local_cache_dir="./models/cached",
                    num_classes=5,
                    variant="Default",
                )
                candidates = loader.sync_metadata_with_mlflow()
        ```
    '''

    def __init__(self,registered_model_name:str,
                cfg:RootConfig,
                local_cache: bool = True,
                local_cache_dir: Optional[str| Path] = None,
                num_classes: int = 5,cache_metadata: bool = True, 
                model_top_n: int = 5,
                metadata: Optional[Metadata]= None,
                metadata_save_file_path: Optional[str|Path] = None,
                metric:str = "f1_macro",
                other_metrics: Optional[list[str]] = None,
                variant: Optional[Literal["Default","Hubble","Dirlichlet"]] = "Default",
                checksum_method: Literal["sha256","md5","sha1"] = "sha256",
                checksum: bool = True):
        """
        Initialize the ModelLoader.

        Args:
            registered_model_name: The name of the registered model in MLflow.
            local_cache: Whether to use local caching.
            local_cache_dir: The directory for local caching.
            num_classes: The number of classes for the model.
            cache_metadata: Whether to cache metadata about the model. Recommended to be True for better validation and tracking of models.
            metadata: An instance of Metadata to manage model metadata. If None, a new Metadata instance will be created.
            metadata_save_path: The path to save the metadata.
            metric: The primary metric to use for sorting and ranking models. Default is "f1_macro".
            other_metrics: A list of other metrics to track for the model.
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
        # Default metadata path is under the repository root: /models/cached/metadata.json
        default_metadata_path = root / "models" / "cached" / "metadata.json"
        self.metadata_save_path = Path(metadata_save_file_path) if metadata_save_file_path else default_metadata_path
        self.best_run_id = None  # Store the best run ID for later use
        self.variant = variant  # Store the variant for metadata
        self.metric = metric  # Store the metric for sorting runs
        self.other_metrics = other_metrics if other_metrics else []  # Store any additional metrics for metadata
        self.checksum_method = checksum_method  # Store the checksum method for metadata
        self.checksum = checksum
        self.top_n = model_top_n  # Store the top_n for metadata
        self.device = cfg.env.device  # Store the device for model loading


        # consctructor conditions checks and creating files/dirs if needed       
        if self.cache_metadata:

            logger.info(f"Cache metadata is enabled. Metadata will be saved to: {self.metadata_save_path}")

            path = Path(self.metadata_save_path).resolve()

            logger.info(f"Resolved metadata save path: {path}")
            if path is not None:
                if path.exists():
                    if not path.is_file():
                        raise FileNotFoundError(f"{path} is not a file")
                else:
                    logger.warning("Metadata file does not exist. Creating a new metadata file.")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch(exist_ok=True)
                    logger.success(f"Created metadata file at {path}")

            if metadata is not None:
                self.metadata = metadata   

            elif self.metadata_save_path is not None:
                self.metadata = Metadata(save_file_path=self.metadata_save_path)
            else:
                raise ValueError("metadata_save_file_path must be set if cache_metadata=True and no metadata instance is passed.")


        if self.local_cache and self.local_cache_dir:
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_metadata and not variant:
            logger.error("Variant must be provided if cache_metadata is True.")
            raise ValueError("Variant must be provided if cache_metadata is True.")

 

    def compute_checksum(self,run_id:str)->str:
        '''
        Compute the checksum of the model file for a given run_id using the specified checksum method.
        Args:
            run_id: The MLflow run ID for which to compute the checksum.

        ```python
            loader = ModelLoader(registered_model_name="my_model", local_cache=True,
                                local_cache_dir="./cache", num_classes=5)
            checksum = loader.compute_checksum(run_id="e3b866df7f01417fb0db72e0da051f23")
            print(checksum)
            >>> "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3..."
        ```
        '''

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
            
  
    
    def load_with_fallback(self) -> torch.nn.Module:

        '''Load the best model for the given experiment name, with a fallback to other models if best model fails.
       
        Returns:
            The loaded PyTorch model.
        
        Raises:
            RuntimeError: If no model could be loaded from the top N runs.
        
        Example:
            ```python
            loader = ModelLoader(registered_model_name="my_model",top_n=5,local_cache=True, local_cache_dir="./cache", num_classes=5)
            model = loader.load_with_fallback()
            ```
        '''

        if self.cache_metadata:
            self.sync_metadata_with_mlflow()  # Sync metadata with MLflow before loading
        try:
            #try with mlflow to load top n 
            model_list = self._best_run(self.registered_model_name,top_n=self.top_n, metric=self.metric)

            #self.metadata.update_metadata()
        except Exception as e:

            #fallback to locl metadata if mlflow is unreachable
            logger.error(f"MLflow unreachable, attempting to load from local metadata fallback: {e}")
            return self._load_from_local_metadata()    

        
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
                logger.debug(f"Failed to load model from run '{run_id}' with {metric}: {metric} (Rank: {rank}). Error: {e}")
                logger.error(f"Couldn't load the model from run_id:{run_id} continuing search")
                continue
        
        raise RuntimeError(f"Failed to load any model for experiment '{self.registered_model_name}' after checking top {self.top_n} runs.")
   
        

    def load_champion_model(self)->Optional[torch.nn.Module]:

        ''' load the champion model from MLflow using the alias "champion". 
        If the alias is not found, it falls back to the latest run in the metadata for the specified variant.

        Returns:
            The loaded PyTorch model if successful, None otherwise.
        
        Example:
        ```python
            loader = ModelLoader(registered_model_name="galaxy-classification-default",
                                local_cache=True, local_cache_dir="./cache",
                                num_classes=5, variant="Default")
            model = loader.load_champion_model()
            if model is not None:
                model.eval()
        ```
        
        '''

        try:

            #gaurd, try with mlflow fallabck to local metadata
            try:
                logger.info(f"Attempting to load champion model for '{self.registered_model_name}' from MLflow alias 'champion'")
                champion_version = self.client.get_model_version_by_alias(self.registered_model_name, "champion")
                champion_run_id = champion_version.run_id

            except Exception:
                logger.warning("Failed to load from MLflow alias; using local metadata to get run_id")
                met = self.metadata
                file = met._load()
                champion_run_id = file[self.variant].candidates[0].run_id #first run id from the variant metadata

            run = self.client.get_run(champion_run_id)
            

            other_metrics = {metric: run.data.metrics.get(metric, None) for metric in self.other_metrics}
            metric = {self.metric: run.data.metrics.get(self.metric, None)}

            logger.success(f"Champion run ID: {champion_run_id}")

        
            if self.cache_metadata and self.local_cache_dir:
                met = CandidateMetadata(
                    rank=1,
                    run_id=champion_run_id,
                    model_name=self.registered_model_name,
                    experiment_name=self.registered_model_name,
                    metric=metric,#dafulat f1_macro
                    num_classes=self.num_classes,
                    cache_file=str(self._cache_path(champion_run_id)),
                    cached_at=datetime.now(timezone.utc).isoformat(),
                    save_path=str(self.local_cache_dir),
                    other_metrics=other_metrics
                )
                self.metadata.update_metadata(met,metric = self.metric ,variant=self.variant)

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
        Example:
        ```python
            loader = ModelLoader(registered_model_name="my_model", local_cache=True,
                                local_cache_dir="./cache", num_classes=5)
            model = loader.check_cache(run_id="e3b866df7f01417fb0db72e0da051f23")
        ```
        
        '''

        logger.info(f"Checking cache for run '{run_id}'")

        cache_path = self._cache_path(run_id)

        #local cache hit
        if self.local_cache and self.local_cache_dir and cache_path.exists():
            if not self.checksum or self.compare_checksum(run_id):
                logger.info(f"Loading state_dict from cache: {cache_path}")
                state_dict = torch.load(cache_path, map_location=self.device, weights_only=True)
                model = self._init_model()
                model.load_state_dict(state_dict)
                return model
            
            else:
                logger.warning(f"Checksum mismatch for '{run_id}'; deleting corrupt cache and re-pulling")
                cache_path.unlink(missing_ok=True)
                self._checksum_path(run_id).unlink(missing_ok=True)
                # falls through to MLflow pull below
            

        logger.info(f"Cache miss for run '{run_id}', pulling from MLflow")

        #load form mlflow then save
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
                    added_checksum = self.compute_checksum(run_id)
                    self._save_checksum(run_id, added_checksum)#save checksum to save dir 
                    logger.info(f"Checksum ({self.checksum_method}) for run '{run_id}': {added_checksum}")

        except Exception as e:
            logger.exception(f"Failed to load model from MLflow for run '{run_id}': {e}")
            #azureblobfallback(run_id)
            raise e

        return model  

    def _load_from_local_metadata(self) -> torch.nn.Module:

        '''Used only when MLflow is unreachable. Reads previously-cached, already-ranked
        candidates from the local metadata file.
        
       
        Example:
        ```python
            # called internally by load_with_fallback when MLflow is unreachable
            model = loader._load_from_local_metadata()
        ```

        '''

        if not (self.cache_metadata and self.metadata_save_path and self.metadata_save_path.exists()):
            raise RuntimeError("MLflow unreachable and no local metadata available to fall back on.")

        file = self.metadata._load()

        candidates = file[self.variant].candidates[:self.top_n]

        if not candidates:
            raise RuntimeError(f"MLflow unreachable and no cached candidates for variant '{self.variant}'.")

        for c in candidates:
            logger.info(f"[local fallback] Attempting run '{c.run_id}' (rank {c.rank}, {self.metric}={c.metric.get(self.metric, None)})")
            try:
                return self.check_cache(c.run_id)
            
            except Exception:
                logger.error(f"[local fallback] Couldn't load run_id:{c.run_id}, continuing")
                continue

        raise RuntimeError(f"Failed to load any model for variant '{self.variant}' from local metadata fallback.")

    
    def _save_checksum(self, run_id: str, checksum: str,save_path:Optional[str|Path]=None) -> None:

        '''Save the checksum for a given run_id to a file in the local cache directory.
        Args:
            run_id: The MLflow run ID for which to save the checksum.
            checksum: The checksum to save.
            save_path: Optional; if provided, the path to the directory where the checksum file will be saved. If not provided, the default local cache directory is used.
        
        Example:
        ```python
            checksum = loader.compute_checksum(run_id="e3b866df7f01417fb0db72e0da051f23")
            loader._save_checksum(run_id="e3b866df7f01417fb0db72e0da051f23", checksum=checksum)
        ```
        '''

        if save_path:
            path = Path(save_path)
        else:
            path = self.local_cache_dir
        if not self.local_cache_dir:
            logger.error("Local cache directory is not set; cannot save checksum.")
            raise

        checksum_path = self._checksum_path(run_id)
        checksum_path.write_text(checksum)
        logger.success(f"Checksum saved to {checksum_path}")

      
    def compare_checksum(self,run_id:str,checksum:Optional[str]=None)->bool:

        '''
        Compared two checksum based on value provided or checks local cache directory for the checksum file of the model using run_id.
        Args:
            run_id: The MLflow run ID for which to compare the checksum.
            checksum: Optional; if provided, compares this checksum with the stored one. If not provided, reads the stored checksum from the local cache.
        Returns:
            True if the checksums match, False otherwise.
        
        Example:
            ```python
            loader = ModelLoader(registered_model_name="my_model",
                                 local_cache=True, local_cache_dir="./cache", num_classes=5)

            is_valid = loader.compare_checksum(run_id="12345")
            print(f"Checksum valid: {is_valid}")
            >>> True
            ```
        '''

        checksum_path = self._checksum_path(run_id)
        if not checksum_path.exists():
            logger.warning(f"No checksum file for run '{run_id}'; treating cache as unverifiable")
            return False
        if checksum is None:
            checksum = self.compute_checksum(run_id)
        return checksum_path.read_text().strip() == checksum
    
    def sync_metadata_with_mlflow(self)-> list[CandidateMetadata]:

        '''Query MLflow for the top_n runs by metric and record all of them in local
        metadata, regardless of whether their weights get downloaded/cached. Keeps the
        local registry in sync with MLflow even for candidates load_with_fallback never
        had to fall through to.


        Returns:
            The list of CandidateMetadata written, in rank order. Empty list if
            cache_metadata is disabled or no runs were found.

        Example:
        ```python
            loader = ModelLoader(registered_model_name="galaxy-classification-default",
                                local_cache=True, local_cache_dir="./cache",
                                num_classes=5, variant="Default",top_n=5)
            candidates = loader.sync_metadata_with_mlflow()
            for c in candidates:
                print(c.rank, c.run_id, c.val_f1_macro)
        ```
        '''

        if not self.cache_metadata:
            logger.warning("cache_metadata is False; skipping metadata sync.")
            return []

        model_list = self._best_run(self.registered_model_name, top_n=self.top_n, metric=self.metric)
        if not model_list:
            logger.warning(f"No runs found for '{self.registered_model_name}' to sync metadata from.")
            return []

        synced = []

        logger.info(f"Syncing top {len(model_list)} runs for '{self.registered_model_name}' to local metadata")

        for rank, run in enumerate(model_list, start=1):
            run_id = run.info.run_id
            cache_path = self._cache_path(run_id) if self.local_cache_dir else None
            other_metrics = {metric: run.data.metrics.get(metric, None) for metric in self.other_metrics}
            metric = {self.metric: run.data.metrics.get(self.metric, None)}

            cand = CandidateMetadata(
                rank=rank,
                run_id=run_id,
                model_name=self.registered_model_name,
                experiment_name=self.registered_model_name,
                metric=metric,
                num_classes=self.num_classes,
                cache_file=str(cache_path) if cache_path else None,
                # only stamp cached_at if the weights are actually on disk already
                cached_at=datetime.now(timezone.utc).isoformat(),
                save_path=str(self.local_cache_dir) if self.local_cache_dir else None,
                other_metrics=other_metrics
            )
            self.metadata.update_metadata(cand, metric=self.metric, variant=self.variant, top_n=self.top_n)# save synced to metatada
            synced.append(cand)

        logger.success(f"Synced top {len(synced)} candidate(s) metadata for '{self.registered_model_name}'")
        return synced

    def _best_run(self,model_name:str,top_n:int=5, metric: str = "f1_macro"):
        '''Retrieve the best run based on the highest specified metric.

        Args:
            model_name: The name of the model in Mlflow Registry.
            top_n: The number of top runs to consider.
            metric: The metric to sort runs by.

        Returns:
            A list of the top N runs based on the specified metric.

        Example:
        ```python
            runs = loader._best_run(model_name="galaxy-classification-default-experiments",
                                    top_n=5, metric="f1_macro")
            for r in runs:
                print(r.info.run_id, r.data.metrics.get("f1_macro"))
        ```

        '''
        logger.info(f"Searching for top {top_n} runs for model '{model_name}' sorted by metric '{metric}'")
        versions = self.client.search_model_versions(f"name='{model_name}'")

        if not versions:
            logger.error(f"No registered model found with name '{model_name}'")
            return None

        #query all the experment ids for the model versions, as experment ids can be independent to model 
        experiment_ids = {
            self.client.get_run(v.run_id).info.experiment_id
            for v in versions
        }


        if experiment_ids is None:
            logger.error(f"No model found with name '{model_name}'")
            return None

        runs = self.client.search_runs(
            experiment_ids=list(experiment_ids),
            filter_string="attributes.status = 'FINISHED'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n,
            )
        logger.success(f"Found {len(runs)} finished runs for model '{model_name}' sorted by metric '{metric}'")

        if not runs: 
            logger.error(f"No finished runs found for experiment '{model_name}'")
            return None
        return runs



    def _checksum_path(self, run_id: str, save_path: Optional[str | Path] = None) -> Path:
    
        '''
        Helper for checksum path. That depends on _cache_path and the checksum method. 
        It returns the path to the checksum file for a given run_id.
        
        Args:
            run_id: The MLflow run ID for which to compute the checksum path.
            save_path: Optional; if provided, the path to the directory where the checksum file will be saved. If not provided, the default local cache directory is used.
        Example:
            ```python
            >>> modelloader._checksum_path("12345",path="./path/to/metadata/12345.sha256")
            PosixPath('path/to/metadata/12345.sha256')
        '''
        if not self.local_cache_dir:
            logger.error("Local cache directory is not set; cannot compute checksum path.")
            raise
        if save_path:
            path = Path(save_path)
        else:
            path = self._cache_path(run_id)

        return path.with_suffix(f".{self.checksum_method}")

    def _init_model(self) -> torch.nn.Module:
        
        from src.models.model import GalaxyClassifier_ResNet
        return GalaxyClassifier_ResNet(N=2, k=2, num_classes=self.num_classes)# for proper mapping
    
    def _cache_path(self, run_id: str) -> Path:

        '''Compute the cache path for a given run_id.
        
        Args:
            run_id: The MLflow run ID for which to compute the cache path.
        
        Example:
        ```python
            >>> loader._cache_path("e3b866df7f01417fb0db72e0da051f23")
            PosixPath('models/cached/e3b866df7f01417fb0db72e0da051f23.pt')
        ```

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

class ModelEvictor(Metadata):
    pass


