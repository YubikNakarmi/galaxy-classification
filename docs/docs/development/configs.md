# ML configuration

The ML pipeline reads [`configs/ml.yaml`](../../../configs/ml.yaml) when the
CLI context is created. `configs/ml_kaggle.yaml` is an alternative profile for
Kaggle paths and must be selected explicitly by tooling or copied to
`configs/ml.yaml`; the CLI does not switch profiles automatically.

The YAML is parsed into the typed models in
[`src/config/schema.py`](../../../src/config/schema.py). Relative paths are
resolved from the repository root.

## Configuration sections

| Section | Purpose |
| --- | --- |
| `active_variant` | Variant used by the loaded configuration. |
| `env` | Seed, device, and dataloader workers. |
| `mode` | `local` or `cloud` execution mode. |
| `variants` | Per-dataset split, model, experiment, and MLflow settings. |
| `data` | Image, label, split, normalization, transform, and sampler settings. |
| `storage` | Local directories and Azure Blob paths for artifacts. |
| `augmentation` | Reserved augmentation settings. |
| `train` | Optimizer, scheduler, checkpoint, and MLflow logging settings. |
| `eval` | Evaluation batch size and metrics. |
| `mlflow` | Tracking backend and artifact-store metadata. |

Inspect the resolved configuration before a run:

```bash
galax pipeline train --status
```

The command displays the configuration and exits without training.

## Environment

```yaml
env:
  name: galaxy_classification_env
  python_version: 3.12.10
  seed: 42
  device: cpu
  num_workers: 4
```

`device` can be `cpu`, `cuda`, or a CUDA device such as `cuda:0`. Set it to
`cuda` only when the selected PyTorch installation and host provide a usable
GPU.

## Variants

`active_variant` must match one of the keys under `variants`:

```yaml
active_variant: Default

variants:
  Default:
    preprocessing: Default
    num_classes: 5
    label_file: data/preprocessed/labels/labeled_data_1.csv
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
    experiment:
      root: galaxy-classification-default-experiments
      auto_timestamp: true
      use_variant_name: true
    mlflow:
      model_name: galaxy-classification-default
```

The supported variant names are `Default`, `Hubble`, and `Dirichlet`.
`dirichlet_branches` is used by variants that provide Dirichlet outputs.
The `train` command currently executes the `Default` training implementation;
the other variant entry points are reserved for future implementation.

## Data

```yaml
data:
  img_dir: images/
  raw_paths:
    - labels: data/raw/labels.csv
    - mappings: data/raw/mapping.csv
  preprocessed_dir: data/preprocessed/
  split_dir: data/preprocessed/split/
  train_path: data/preprocessed/split/train.csv
  val_path: data/preprocessed/split/val.csv
  eval_path: data/preprocessed/split/test.csv
  input_size: 128
  normalize:
    mean: [0.1376, 0.1276, 0.1187]
    std: [0.1728, 0.1586, 0.1488]
  sampler:
    type: weighted
```

`train_path` and `val_path` are required for training. `eval_path` is used by
evaluation. The `raw_paths` entries are consumed by preprocessing and splitting.

## Training

```yaml
train:
  batch_size: 64
  num_epochs: 70
  learning_rate: 0.001
  weight_decay: 0.01
  optimizer: AdamW
  grad_clip_max_norm: 1.0
  label_smoothing: 0.1
  criterion:
    name: cross_entropy
    weight: true
  scheduler:
    type: cosine_warmup
    warmup_epochs: 10
    min_lr: 0.000001
  regularization:
    dropout: 0.2
    gradient_clipping: true
```

The scheduler type is `cosine_warmup` or `cosine_warmup_restarts`. Restart
schedulers may also define `T_0` and `T_mult`.

### Checkpoints and resume

```yaml
train:
  checkpoints:
    resume_from_checkpoint: false
    local_checkpoint_path: models/checkpoints
    checkpoint_restore_path: checkpoints
    resume_run_id: null
    checkpoint_resume_hard: true
  checkpoint:
    save_freq_epoch: 10
    save_cycle: true
    save_latest: true
```

`checkpoints` controls restoring a previous run. `checkpoint` controls periodic
saves. The CLI flags `--resume`, `--resume_run_id`, and
`--resume_checkpoint_from Best|Latest` override the corresponding run-time
choices. A hard resume fails when the requested checkpoint cannot be found;
otherwise training starts a new run.

## Storage and MLflow

Local paths are configured under `storage.local`:

```yaml
storage:
  local:
    logs:
      dir: logs/
      tensorboard_dir: logs/tensorboard/
      system_dir: logs/system/
      reports_dir: logs/reports/
    checkpoints:
      dir: models/checkpoints/
      model_cache_dir: models/cache/
    predictions:
      dir: models/predictions/
```

Cloud destinations are under `storage.cloud`, including the DVC, experiments,
logs, tensorboard, checkpoints, and predictions containers. Azure authentication
comes from `AZURE_STORAGE_CONNECTION_STRING`.

Training MLflow logging is controlled by `train.mlflow`:

```yaml
train:
  mlflow:
    log_data: true
    log_model: true
    log_metrics: true
    experiment_name: galaxy-classification-training
```

Set `MLFLOW_TRACKING_URI` in `.env` to select the tracking server. The CLI
defaults to `http://localhost:5000` when it is not set.
