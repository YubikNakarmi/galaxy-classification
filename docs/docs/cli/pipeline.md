# ML pipeline commands

All ML pipeline commands are below `galax pipeline`:

```bash
galax pipeline --help
```

The group exposes `preprocess`, `train`, `evaluate`, and `load-metrics`.

## `pipeline preprocess`

```bash
galax pipeline preprocess [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run_preprocessing BOOLEAN` | `False` | Generate processed labels from configured raw files. |
| `--preprocess_type TEXT` | `classic` | Preprocessing strategy. |
| `--run_split BOOLEAN` | `False` | Write train, validation, and test CSV splits. |
| `--variant [Default\|Hubble\|Dirichlet]` | `Default` | Variant whose data configuration is used. |

```bash
galax pipeline preprocess \
  --run_preprocessing true \
  --preprocess_type classic \
  --run_split true \
  --variant Default
```

The command reads raw mapping and label paths from the active configuration.
Unsupported preprocessing types are skipped with a warning.

## `pipeline train`

See the [training and evaluation workflow](train_and_eval.md) for a complete
runbook.

```bash
galax pipeline train [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--variant [Default\|Hubble\|Dirichlet]` | `-v` | `Default` | Training variant. |
| `--evaluate` | `-e` | off | Evaluate the model after training. |
| `--resume` | `-r` | off | Resume from the available checkpoint. |
| `--status` | `-c` | off | Show configuration and exit. |
| `--yes` | `-y` | off | Skip confirmation prompts. |
| `--resume_run_id TEXT` | `-rid` | — | Checkpoint run ID to resume. |
| `--resume_checkpoint_from [Best\|Latest]` | — | `Best` | Checkpoint to restore. |

```bash
galax pipeline train --status
galax -vv pipeline train --variant Default --yes
galax pipeline train --variant Default --resume --resume_run_id RUN_ID
```

`--status` exits after displaying configuration. `--resume` uses the
checkpoint configuration, while `--resume_run_id` selects a specific parent
run.

## `pipeline evaluate`

```bash
galax pipeline evaluate [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run_id TEXT` | **required** | MLflow run ID to evaluate. |
| `--variant [Default\|Hubble\|Dirichlet]` | `Default` | Model variant. |
| `--model_version TEXT` | — | MLflow registry version. |
| `--min-improvement FLOAT` | `0.005` | Improvement required for promotion. |
| `--metric_name TEXT` | `f1_macro` | Metric used for promotion. |
| `--model_source [mlflow\|local]` | `mlflow` | Load from MLflow or a local checkpoint. |
| `-m PATH` | — | Local model path when using `local`. |
| `--alias TEXT` | `Champion` | Registry alias to update. |
| `--register` | enabled | Register a missing model in MLflow. |

```bash
galax pipeline evaluate --run_id RUN_ID --variant Default
galax pipeline evaluate --run_id local-check \
  --model_source local \
  -m models/checkpoint.pt
```

When `--model_source local` is used, `-m` is required. When the MLflow source
is used, omit the local path. `--model_version` selects a registered MLflow
model version.

## `pipeline load-metrics`

```bash
galax pipeline load-metrics [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--run_id TEXT` | `-r` | **required** | Training run whose metrics are loaded. |
| `--save_path PATH` | `-s` | — | CSV file or directory for combined metrics. |
| `--metric_names TEXT` | `-m` | all | Metric to include; repeat for multiple metrics. |
| `--exclude_system` | `-e` | enabled | Exclude MLflow system metrics. |

```bash
galax pipeline load-metrics -r RUN_ID \
  -m accuracy -m f1_macro \
  -s reports/metrics.csv
```
