# Ml.yaml


## Overview

The configuration file is divided into the following sections:

| Section | Purpose |
|---|---|
| `env` | Runtime environment settings |
| `experiment` | Experiment naming and artifact organization |
| `active_variant` | Selects the active experiment variant |
| `variants` | Task-specific configuration overrides |
| `data` | Dataset locations and preprocessing |
| `storage` | Local and cloud storage locations |
| `augmentation` | Data augmentation settings |
| `model` | Model architecture defaults |
| `train` | Optimization and learning settings |
| `eval` | Evaluation behavior and metrics |
| `mlflow` | Experiment tracking integration |

Each section controls a specific part of the machine learning pipeline.

---

## Environment Configuration

Controls runtime settings used throughout the project.

```yaml
env:
  name: galaxy_classification_env
  python_version: 3.14.4
  seed: 42
  device: cpu
  num_workers: 4
```

| Parameter | Description |
|---|---|
| `name` | Name of the Python environment. |
| `python_version` | Python version used for development and training. |
| `seed` | Global random seed for reproducibility. |
| `device` | Compute device (`cpu` or `cuda`). |
| `num_workers` | Number of dataloader worker processes. |

---

## Active Variant

Selects which experiment variant is currently active.

```yaml
active_variant: Default
```

The training pipeline loads configuration values from the selected variant.

!!! example
    ```yaml
    active_variant: Hubble
    ```

---

## Experiment Configuration

Controls experiment naming and artifact organization.

```yaml
experiment:
  root: galaxy-classification-experiments
  auto_timestamp: true
  use_variant_name: true
```

| Parameter | Description |
|---|---|
| `root` | Root directory for experiment artifacts. |
| `auto_timestamp` | Appends a timestamp to experiment names. |
| `use_variant_name` | Includes the variant name in the experiment directory. |

### Generated Experiment Names

When both `auto_timestamp` and `use_variant_name` are enabled:

```
Default_20260605_153000
Hubble_20260605_160412
Dirichlet_20260605_163014
```

---

## Variants

Variants define task-specific configurations.

```yaml
variants:
  Default:
    ...
```

Each variant can override:

- Dataset configuration
- Model architecture
- Label source
- MLflow model name
- Task-specific settings

### Default Variant

Used for standard galaxy morphology classification.

```yaml
Default:
  num_classes: 5

  data:
    split:
      train_size: 0.70
      val_size: 0.15
      test_size: 0.15
    stratify: true

  model:
    architecture: ResNet
    pretrained: false
    K: 2
    N: 2
    conv1_stride: 2
    input_size: 128
```

**Model parameters:**

| Parameter | Description |
|---|---|
| `K` | Width multiplier for custom ResNet. |
| `N` | Number of residual blocks. |
| `conv1_stride` | Initial convolution stride. |
| `input_size` | Image resolution. |

### Hubble Variant

Used for Galaxy Zoo vote fraction prediction.

```yaml
Hubble:
  num_classes: 10
  dirichlet_branches: 11
```

The model creates multiple prediction heads for vote fraction estimation:

```
Input Features
      │
 ┌────┼────┐
 │    │    │
Head1 Head2 HeadN
```

### Dirichlet Variant

Used for probabilistic regression tasks.

```yaml
Dirichlet:
  num_classes: null
```

!!! note
    Unlike classification variants, labels represent probability distributions rather than discrete classes.

---

## Dataset Configuration

Defines dataset locations and preprocessing settings.

```yaml
data:
  raw_dir: data/raw/
  preprocessed_dir: data/preprocessed/
  split_dir: data/preprocessed/split/
```

### Directory Structure

```
data/
├── raw/
├── preprocessed/
│   ├── labels/
│   └── split/
```

### Image Normalization

Dataset normalization statistics applied via `transforms.Normalize(mean, std)`:

```yaml
normalize:
  mean: [0.1376, 0.1276, 0.1187]
  std:  [0.1728, 0.1586, 0.1488]
```

### Sampling Strategy

```yaml
sampler:
  type: weighted
```

| Type | Description |
|---|---|
| `weighted` | Handles class imbalance using weighted sampling. |
| `random` | Uniform random sampling. |

---

## Storage Configuration

Defines local and cloud storage locations.

### DVC Storage

Used for dataset versioning.

```yaml
storage:
  dvc:
    container: galaxy-dvc
```

```
galaxy-dvc/
└── DVC-managed datasets
```

Recommended for raw datasets, processed datasets, and data splits.

### Experiment Storage

Used for training artifacts.

```yaml
storage:
  experiments:
    container: galaxy-experiments
```

```
galaxy-experiments/
├── checkpoints/
├── logs/
├── tensorboard/
└── predictions/
```

### Local Storage

Training writes artifacts locally before cloud upload.

```yaml
storage:
  local:
    logs:
      dir: logs/
```

```
project/
├── logs/
├── checkpoints/
└── predictions/
```

### Cloud Upload Configuration

```yaml
cloud:
  upload:
    checkpoint_every_epoch: true
    best_model_only: true
    logs_every_n_epochs: 5
```

| Parameter | Description |
|---|---|
| `checkpoint_every_epoch` | Upload latest checkpoint after every epoch. |
| `best_model_only` | Upload only the best validation model. |
| `logs_every_n_epochs` | Upload logs periodically every N epochs. |

---

## Model Configuration

Defines default model settings.

```yaml
model:
  architecture: Default
  pretrained: false
  num_classes: 5
  input_size: 128
```

| Parameter | Description |
|---|---|
| `architecture` | Model architecture name. |
| `pretrained` | Load pretrained weights. |
| `num_classes` | Number of output classes. |
| `input_size` | Input image size (pixels). |

---

## Training Configuration

Controls optimization and learning behavior.

```yaml
train:
  batch_size: 32
  learning_rate: 0.001
  optimizer: AdamW
  grad_clip_max_norm: 1.0
  label_smoothing: 0.1

  criterion:
    name: cross_entropy
    weight: true

  scheduler:
    type: cosine_with_warmup
    warmup_epochs: 5
    min_lr: 0.00001

  regularization:
    dropout: 0.2
    weight_decay: 0.0001
```

### Parameters

| Parameter | Description |
|---|---|
| `batch_size` | Number of samples processed per optimization step. |
| `learning_rate` | Initial optimizer learning rate. |
| `optimizer` | Optimizer type (`AdamW`, `Adam`, `SGD`). |
| `grad_clip_max_norm` | Max gradient norm for clipping — prevents exploding gradients. |
| `label_smoothing` | Softens target labels to reduce overconfidence. |

### Loss Function

| Parameter | Description |
|---|---|
| `name` | Loss function (e.g. `cross_entropy`). |
| `weight` | Apply class weighting to the loss. |

### Learning Rate Scheduler

The `cosine_with_warmup` scheduler follows this workflow:

```
Warmup (warmup_epochs)
        │
        ▼
    Peak LR
        │
        ▼
  Cosine Decay → min_lr
```

### Regularization

| Parameter | Description |
|---|---|
| `dropout` | Dropout probability applied during training. |
| `weight_decay` | L2 regularization coefficient. |

---

## Checkpointing

Controls model persistence during training.

```yaml
checkpoint:
  save_freq_epoch: 10
  save_best: true
  save_latest: true
```

### Example Output

```
checkpoints/
├── latest.pt
├── best.pt
├── epoch_10.pt
├── epoch_20.pt
└── epoch_30.pt
```

---

## Evaluation Configuration

Controls evaluation behavior and reported metrics.

```yaml
eval:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
```

| Metric | Description |
|---|---|
| `accuracy` | Overall classification correctness. |
| `precision` | Positive prediction quality. |
| `recall` | Sensitivity to positive samples. |
| `f1_score` | Harmonic mean of precision and recall. |

---

## MLflow Configuration

Controls experiment tracking integration.

```yaml
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}
  log_data: true
  log_model: true
  log_metrics: true
```

| Parameter | Description |
|---|---|
| `tracking_uri` | MLflow server URI (resolved from environment variable). |
| `backend` | Backend database type. |
| `artifact_store` | Artifact storage backend. |
| `experiment_name` | MLflow experiment name. |

### Logged Artifacts

When enabled, the training pipeline automatically logs:

- Training and validation metrics
- Hyperparameters
- Model checkpoints
- TensorBoard logs
- Predictions

---

## Recommended Storage Architecture

```
Storage Account
├── galaxy-dvc
│   └── datasets
│
└── galaxy-experiments
    └── Default_20260605_153000
        ├── checkpoints/
        ├── logs/
        ├── tensorboard/
        └── predictions/
```

!!! tip
    This separation keeps datasets versioned through DVC while storing experiment artifacts independently, enabling efficient experiment management without polluting your data version history.