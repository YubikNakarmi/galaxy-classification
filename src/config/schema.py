
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvConfig(ConfigModel):
    name: str
    python_version: str
    seed: int
    device: str
    num_workers: int


class SplitConfig(ConfigModel):
    train_size: float
    val_size: float
    test_size: float


class VariantDataConfig(ConfigModel):
    split: SplitConfig
    stratify: bool


class ExperimentConfig(ConfigModel):
    root: str
    auto_timestamp: bool
    use_variant_name: bool


class ModelConfig(ConfigModel):
    architecture: str
    pretrained: bool
    K: int
    N: int
    conv1_stride: int
    input_size: int


class VariantMLflowConfig(ConfigModel):
    model_name: str


class VariantConfig(ConfigModel):
    name: str
    preprocessing: str
    num_classes: int | None
    label_file: Path
    data: VariantDataConfig
    experiment: ExperimentConfig
    mlflow: VariantMLflowConfig
    model: ModelConfig | None = None
    dirichlet_branches: int | None = None


class NormalizeConfig(ConfigModel):
    mean: list[float]
    std: list[float]


class RawDataConfig(ConfigModel):
    labels: Path
    mappings: Path


class TransformConfig(ConfigModel):
    raw_transforms: list[dict[str, dict[str, Any]]]


class SamplerConfig(ConfigModel):
    type: str


class DataConfig(ConfigModel):
    img_dir: Path
    raw_paths: RawDataConfig
    preprocessed_dir: Path
    split_dir: Path
    train_path: Path
    val_path: Path
    eval_path: Path
    input_size: int
    normalize: NormalizeConfig
    transforms: TransformConfig
    sampler: SamplerConfig


class LocalLogsConfig(ConfigModel):
    dir: Path
    tensorboard_dir: Path
    system_dir: Path
    reports_dir: Path


class LocalCheckpointsConfig(ConfigModel):
    dir: Path
    model_cache_dir: Path


class LocalPredictionsConfig(ConfigModel):
    dir: Path


class LocalStorageConfig(ConfigModel):
    logs: LocalLogsConfig
    checkpoints: LocalCheckpointsConfig
    predictions: LocalPredictionsConfig


class CloudContainerConfig(ConfigModel):
    container: str


class CloudLogsConfig(ConfigModel):
    container: str
    dir: str


class CloudTensorboardConfig(ConfigModel):
    dir: str


class CloudCheckpointsConfig(ConfigModel):
    container: str
    dir: str


class CloudPredictionsConfig(ConfigModel):
    container: str
    dir: str


class CloudStorageConfig(ConfigModel):
    dvc: CloudContainerConfig
    experiments: CloudContainerConfig
    logs: CloudLogsConfig
    tensorboard: CloudTensorboardConfig
    checkpoints: CloudCheckpointsConfig
    predictions: CloudPredictionsConfig


class StorageConfig(ConfigModel):
    local: LocalStorageConfig
    cloud: CloudStorageConfig


class CheckpointTrainConfig(ConfigModel):
    resume_from_checkpoint: bool
    local_checkpoint_path: Path | None
    checkpoint_restore_path: str | None
    run_id: str | None
    checkpoint_resume_hard: bool


class CriterionConfig(ConfigModel):
    name: str
    weight: bool


class SchedulerConfig(ConfigModel):
    type: str
    warmup_epochs: int
    min_lr: float


class RegularizationConfig(ConfigModel):
    dropout: float
    gradient_clipping: bool


class TrainMLflowConfig(ConfigModel):
    log_data: bool
    log_model: bool
    log_metrics: bool
    experiment_name: str


class TrainCheckpointConfig(ConfigModel):
    save_freq_epoch: int
    save_best: bool
    save_latest: bool


class TrainConfig(ConfigModel):
    checkpoints: CheckpointTrainConfig
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    grad_clip_max_norm: float
    label_smoothing: float
    criterion: CriterionConfig
    scheduler: SchedulerConfig
    regularization: RegularizationConfig
    mlflow: TrainMLflowConfig
    checkpoint: TrainCheckpointConfig


class EvalConfig(ConfigModel):
    batch_size: int
    metrics: list[str]
    checkpoints: dict[str, Any] | None


class MLflowConfig(ConfigModel):
    tracking_uri: str
    backend: str
    artifact_store: str
    experiment_name: str


class RootConfig(ConfigModel):
    active_variant: str
    env: EnvConfig
    mode: str
    variants: dict[str, VariantConfig]
    data: DataConfig
    storage: StorageConfig
    augmentation: dict[str, Any]
    train: TrainConfig
    eval: EvalConfig
    mlflow: MLflowConfig

    @model_validator(mode="after")
    def validate_active_variant(self) -> "RootConfig":
        for name, variant in self.variants.items():
            variant.name = name
        if self.active_variant not in self.variants:
            raise ValueError(f"Unknown active_variant: {self.active_variant}")
        return self

    @property
    def variant(self) -> VariantConfig:
        return self.variants[self.active_variant]

    def resolve_paths(self, root: Path) -> "RootConfig":
        path_fields = [
            (self.data, ("img_dir", "preprocessed_dir", "split_dir", "train_path", "val_path", "eval_path")),
            (self.data.raw_paths, ("labels", "mappings")),
            (self.storage.local.logs, ("dir", "tensorboard_dir", "system_dir", "reports_dir")),
            (self.storage.local.checkpoints, ("dir", "model_cache_dir")),
            (self.storage.local.predictions, ("dir",)),
            (self.train.checkpoints, ("local_checkpoint_path",)),
        ]
        path_fields.extend((variant, ("label_file",)) for variant in self.variants.values())

        for model, fields in path_fields:
            for field_name in fields:
                value = getattr(model, field_name)
                if value is not None and not value.is_absolute():
                    setattr(model, field_name, root / value)
        return self
 