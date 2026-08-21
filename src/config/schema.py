
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional



@dataclass
class EnvConfig:
    name: str
    python_version: str
    seed: int
    device: str
    num_workers: int
 
 
@dataclass
class SplitConfig:
    train_size: float
    val_size: float
    test_size: float
 
 
@dataclass
class VariantDataConfig:
    split: SplitConfig
    stratify: bool
 
 
@dataclass
class ExperimentConfig:
    root: str
    auto_timestamp: bool
    use_variant_name: bool
 
 
@dataclass
class ModelConfig:
    architecture: str
    pretrained: bool
    K: int
    N: int
    conv1_stride: int
    input_size: int
 
 
@dataclass
class VariantMLflowConfig:
    model_name: str
 
 
@dataclass
class VariantConfig:
    name: str                          # Variant name
    preprocessing: str
    num_classes: Optional[int]         # None for Dirichlet
    label_file: Path
    data: VariantDataConfig
    experiment: ExperimentConfig
    mlflow: VariantMLflowConfig
    model: Optional[ModelConfig] = None
    dirichlet_branches: Optional[int] = None
 
 
''' Data and transforms'''
 
@dataclass
class NormalizeConfig:
    mean: list[float]
    std: list[float]

@dataclass
class RawDataConfig:
    labels: Path
    mappings: Path
 
@dataclass
class DataConfig:
    img_dir: Path
    preprocessed_dir: Path
    split_dir: Path
    train_path: Path
    val_path: Path
    eval_path: Path
    input_size: int
    normalize: NormalizeConfig
    raw_paths: RawDataConfig
 
 
'''Storages''' 
@dataclass
class LocalLogsConfig:
    dir: Path
    tensorboard_dir: Path
    system_dir: Path
    reports_dir: Path
 
 
@dataclass
class LocalCheckpointsConfig:
    dir: Path
    model_cache_dir: Path
 
 
@dataclass
class LocalStorageConfig:
    logs: LocalLogsConfig
    checkpoints: LocalCheckpointsConfig
 
 
@dataclass
class CloudContainerConfig:
    container: str
 
 
@dataclass
class CloudStorageConfig:
    dvc: CloudContainerConfig
    experiments: CloudContainerConfig
 
 
@dataclass
class StorageConfig:
    local: LocalStorageConfig
    cloud: CloudStorageConfig
 
 
''' Training and checkpoints '''
 
@dataclass
class CheckpointTrainConfig:
    resume_from_checkpoint: bool
    local_checkpoint_path: Optional[Path]
    checkpoint_restore_path: Optional[str]       # blob path, not a local Path
    checkpoint_resume_hard: bool
    run_id: Optional[str] = None  # Optional run_id for checkpointing, can be None if not specified
 
 
@dataclass
class SchedulerConfig:
    type: str
    warmup_epochs: int
    min_lr: float
 
 
@dataclass
class TrainConfig:
    checkpoints: CheckpointTrainConfig
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    grad_clip_max_norm: float
    label_smoothing: float
    scheduler: SchedulerConfig
 
 
'''Mlflow'''
@dataclass
class MLflowConfig:
    backend: str
    artifact_store: str
    experiment_name: str
 
 
''' root configs'''
@dataclass
class RootConfig:
    env: EnvConfig
    mode: str
    active_variant: str
    variants: dict[str, VariantConfig]
    data: DataConfig
    storage: StorageConfig
    train: TrainConfig
    mlflow: MLflowConfig
 
    # Shortcut: pre-resolved active variant
    @property
    def variant(self) -> VariantConfig:
        return self.variants[self.active_variant]
 