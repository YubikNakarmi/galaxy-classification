# Training loop

The training command is a Click command under `galax pipeline`. It validates
the configured image and split paths, starts an MLflow run, trains the model,
saves checkpoints, and optionally evaluates the completed run.

## Running training

Inspect the resolved configuration first:

```bash
galax pipeline train --status
```

Start a run after preprocessing and splitting have produced the configured
files:

```bash
galax pipeline train --variant Default --yes
```

Omit `--yes` to receive an interactive confirmation. Use `-vv` for verbose
CLI output.

Resume a previous run:

```bash
galax pipeline train \
  --variant Default \
  --resume \
  --resume_run_id RUN_ID \
  --resume_checkpoint_from Best \
  --yes
```

`--resume_checkpoint_from` accepts `Best` or `Latest`. `--evaluate` runs the
evaluation command after training completes.

!!! note
    The `Hubble` and `Dirichlet` names are accepted by the shared CLI and
    configuration schema, but their training entry points are not implemented
    yet. The current training implementation runs `Default`.

## Training configuration

All training values are read from [`configs/ml.yaml`](../../../configs/ml.yaml).
The most commonly changed keys are:

| Key | Purpose |
| --- | --- |
| `env.device` | CPU or CUDA device. |
| `env.seed` | Random seed. |
| `train.batch_size` | Samples per optimization step. |
| `train.num_epochs` | Number of epochs. |
| `train.learning_rate` | Initial learning rate. |
| `train.weight_decay` | AdamW weight decay. |
| `train.grad_clip_max_norm` | Maximum gradient norm. |
| `train.label_smoothing` | Cross-entropy label smoothing. |
| `train.scheduler.*` | Warmup, minimum LR, and cosine scheduler settings. |
| `train.checkpoints.*` | Resume source and hard-resume behavior. |
| `train.checkpoint.*` | Periodic and latest checkpoint saves. |
| `train.mlflow.*` | Data, model, and metric logging. |

See [ML configuration](../development/configs.md) for the complete schema and
storage settings.

## Checkpoints and artifacts

Training writes local artifacts to the configured storage paths and records the
run in MLflow. The checkpoint manager maintains:

- `best.pt` for the best validation result;
- `latest.pt` when `train.checkpoint.save_latest` is enabled;
- periodic epoch or cycle checkpoints when configured;
- reports, TensorBoard events, and MLflow metrics.

The local checkpoint directory is `storage.local.checkpoints.dir`. Resume
downloads use `train.checkpoints.local_checkpoint_path` and the configured
cloud restore path when cloud storage is enabled.

## Evaluation

Evaluate an MLflow run separately:

```bash
galax pipeline evaluate \
  --run_id RUN_ID \
  --variant Default \
  --model_source mlflow
```

For a local checkpoint:

```bash
galax pipeline evaluate \
  --run_id local-check \
  --model_source local \
  --model_path models/checkpoints/best.pt
```

Evaluation uses `eval.batch_size` and the metric names in `eval.metrics`.
Promotion settings such as `--metric_name`, `--min-improvement`, and
`--alias` are documented in the [pipeline command reference](../cli/pipeline.md).

## Training factors

The default profile uses AdamW with cross-entropy loss, optional class
weighting, label smoothing, gradient clipping, and cosine warmup. Adjust these
values in the YAML profile for a specific dataset or hardware target; do not
rely on the older illustrative values from this page.
