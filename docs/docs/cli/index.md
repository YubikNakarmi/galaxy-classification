# Command-line interface

The project installs the `galax` command when the package is installed:

```bash
uv sync
uv run galax --help
```

You can also use an activated virtual environment:

```bash
pip install -e .
galax --help
```

!!! tip
    Add `-v`, `-vv`, or `-vvv` immediately after `galax` to increase diagnostic
    output. Run `galax COMMAND --help` for the options of any command.

## Command tree

| Command | Purpose |
| --- | --- |
| `galax status` | Display project configuration and service status. |
| `galax log show` | Read and filter application or ML logs. |
| `galax pipeline ...` | Preprocess data, train, evaluate, and load MLflow metrics. |
| `galax cache ...` | Import, inspect, export, modify, or delete local models. |
| `galax setup ...` | Create the directories required by the project. |
| `galax cloud ...` | Check Azure Blob Storage or upload a file. |
| `galax bentoml ...` | Build and containerize the BentoML service. |

## Global options

```text
galax [OPTIONS] COMMAND [ARGS]...

Options:
  --version      Show the installed version.
  -v, --verbose  Increase verbosity level (-v, -vv, -vvv).
  -h, --help     Show help and exit.
```

Global options must appear before the command:

```bash
galax -vv pipeline train --status
galax --version
```

Every command and command group also accepts `-h`/`--help`.

## Project status and logs

### `status`

```bash
galax status
```

Displays the resolved configuration, MLflow tracking URI, and project
environment information.

`status` has no project-specific options:

```bash
galax status --help
```

### `log`

`log` is a command group. Use `log show` to read the configured log file:

```bash
galax log --help
galax log show --help
```

### `log show`

```bash
galax log show [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `-t, --type [ml\|app]` | `ml` | Select the ML or application log. |
| `-l, --level [DEBUG\|INFO\|SUCCESS\|WARNING\|ERROR\|CRITICAL]` | — | Keep only one log level. |
| `-n, --lines INTEGER` | `50` | Number of matching lines to display. |
| `-s, --service TEXT` | — | Filter by service name. |

The `--type` value selects either the ML log (`ml`) or application log
(`app`). `--lines` limits the output after the other filters are applied.

For example:

```bash
galax log show --type ml --level ERROR --lines 25
galax log show --service training
```

## ML pipeline

All pipeline commands are below `galax pipeline`.

```bash
galax pipeline --help
```

The pipeline group itself has no options; it exposes `preprocess`, `train`,
`evaluate`, and `load-metrics`.

### `pipeline preprocess`

```bash
galax pipeline preprocess [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--run_preprocessing BOOLEAN` | `False` | Generate processed labels from the configured raw files. |
| `--preprocess_type TEXT` | `classic` | Preprocessing strategy to use. |
| `--run_split BOOLEAN` | `False` | Write train, validation, and test CSV splits. |
| `--variant [Default\|Hubble\|Dirichlet]` | `Default` | Variant whose data configuration is used. |

```bash
galax pipeline preprocess --run_preprocessing true --run_split true
```

The command reads the raw mapping and label paths from the active
configuration. `--preprocess_type` is currently intended for the classic
workflow; unsupported values are skipped with a warning.

### `pipeline train`

See the dedicated [training and evaluation guide](train_and_eval.md).

```bash
galax pipeline train [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--variant` | `-v` | `Default` | Training variant (`Dirlichlet` is the spelling currently accepted by this command). |
| `--evaluate` | `-e` | off | Evaluate after training. |
| `--resume` | `-r` | off | Resume from the available checkpoint. |
| `--status` | `-c` | off | Show the training configuration and exit. |
| `--yes` | `-y` | off | Skip confirmation prompts. |
| `--resume_run_id TEXT` | `-rid` | — | Checkpoint run ID to resume. |
| `--resume_checkpoint_from [Best\|Latest]` | — | `Best` | Checkpoint to restore. |

```bash
galax pipeline train --status
galax -vv pipeline train --variant Default --yes
```

`--status` exits after displaying configuration. `--yes` bypasses the
confirmation prompt. `--resume` uses the checkpoint configuration, while
`--resume_run_id` selects a specific parent run.

### `pipeline evaluate`

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
galax pipeline evaluate --run_id 0123456789abcdef
galax pipeline evaluate --run_id local-check --model_source local \
  -m models/checkpoint.pt
```

The `--register` option is enabled by default when a local model needs to be
registered.

When `--model_source local` is used, `-m` is required. When the
MLflow source is used, omit the local path. `--model_version` takes precedence
when selecting a registered MLflow model.

### `pipeline load-metrics`

```bash
galax pipeline load-metrics [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--run_id TEXT` | `-r` | **required** | Training run whose metrics are loaded. |
| `--save_path PATH` | `-s` | — | CSV file or directory for the combined metrics. |
| `--metric_names TEXT` | `-m` | all | Metric to include; repeat for multiple metrics. |
| `--exclude_system` | `-e` | enabled | Exclude MLflow system metrics. |

Repeat `--metric_names` to request more than one metric:

```bash
galax pipeline load-metrics -r RUN_ID \
  -m accuracy -m f1_macro \
  -s reports/metrics.csv
```

## Model cache

The cache group manages local model files and their metadata:

```bash
galax cache --help
```

### `cache import`

```bash
galax cache import [OPTIONS]
```

Imports a checkpoint into the local model cache and records its metadata.

| Option | Default | Description |
| --- | --- | --- |
| `--checkpoint_path PATH` | **required** | Checkpoint to import. |
| `--model_name TEXT` | **required** | Registered model name. |
| `--variant TEXT` | **required** | `Default`, `Hubble`, or `Dirichlet`. |
| `--num_classes INTEGER` | **required** | Number of output classes. |
| `--run_id TEXT` | generated | Identifier stored with the checkpoint. |
| `--metric_name TEXT` | `vf1_macro` | Metric used to rank the model. |
| `--metric FLOAT` | — | Metric value when it is not in the checkpoint. |
| `--local_cache_dir PATH` | config | Destination for cached weights. |
| `--metadata_save_file_path PATH` | config | Metadata JSON path. |

```bash
galax cache import \
  --checkpoint_path models/checkpoint.pt \
  --model_name galaxy-classification-default \
  --variant Default \
  --num_classes 5 \
  --metric 0.82
```

The checkpoint may be a raw state dictionary or a training
checkpoint containing `model_state_dict`.

### `cache export`

```bash
galax cache export [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--model_path TEXT` | `-p` | **required** | Model file to export. |
| `--export_path TEXT` | `-e` | **required** | Destination file or path. |
| `--input_shape TUPLE` | `-i` | `(1, 3, 128, 128)` | Example input shape used by the exporter. |
| `--format [onnx\|torchscript\|torchscript_pt2\|tensorrt]` | `-f` | **required** | Export format. |

```bash
galax cache export \
  -p models/cache/model.pt \
  -e models/export/model.onnx \
  -f onnx \
  -i "(1,3,128,128)"
```

### Other cache commands

```bash
galax cache show-local-cache [--metadata_path PATH] [--maximum_rows INTEGER]
galax cache export --model_path PATH --export_path PATH \
  --format [onnx|torchscript|torchscript_pt2|tensorrt]
galax cache modify [--run_id TEXT] [--promote]
galax cache delete [OPTIONS]
```

### `cache show-local-cache`

| Option | Default | Description |
| --- | --- | --- |
| `--metadata_path TEXT` | configured metadata path | Metadata JSON to inspect. |
| `--maximum_rows INTEGER` | `5` | Maximum rows requested for display. |

### `cache modify`

| Option | Default | Description |
| --- | --- | --- |
| `--run_id TEXT` | — | Run ID to modify. |
| `--promote` | off | Promote the selected model to champion. |

### `cache delete`

| Option | Default | Description |
| --- | --- | --- |
| `--run_id TEXT` | — | Run ID to delete. |
| `--model_id TEXT` | — | Model ID to delete. |
| `--variant TEXT` | — | Model variant. |
| `--force` | off | Delete without confirmation. |
| `--local_cache_dir TEXT` | configured cache | Local model directory. |
| `--metadata_save_file_path TEXT` | configured metadata | Metadata JSON path. |

Without `--force`, deletion asks for confirmation.

## Setup

`setup` is a command group. It currently exposes `mkdirs`:

```bash
galax setup --help
galax setup mkdirs [--extra_dirs TEXT]
```

`--extra_dirs` is a comma-separated list of additional directories to create.

## Azure cloud storage

The cloud commands require `AZURE_STORAGE_CONNECTION_STRING` in the
environment or `.env` file.

```bash
galax cloud --help
```

### `cloud status`

```bash
galax cloud status
```

Displays the configured Azure Storage account and available container
information. It has no project-specific options.

### `cloud upload-blob`

```bash
galax cloud upload-blob --blob_name results.json --file_path results.json
galax cloud upload-blob --container_name experiments \
  --blob_name reports/run.json --file_path reports/run.json
```

| Option | Short form | Required | Description |
| --- | --- | --- | --- |
| `--container_name TEXT` | — | no | Azure container; uses configured default when omitted. |
| `--blob_name TEXT` | `-b` | yes | Destination blob name. |
| `--file_path TEXT` | `-f` | yes | Local file to upload. |

## BentoML

`bentoml` is a command group:

```bash
galax bentoml --help
galax bentoml containerize [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--bentofile-path PATH` | `-f` | `bentofile.yaml` | BentoML configuration file or directory. |
| `--tag TEXT` | `-t` | `latest` | Docker image version tag. |

```bash
galax bentoml containerize --bentofile-path bentofile.yaml --tag latest
```

The command invokes the BentoML build/containerization process locally, so the
`bentoml` executable and any required Docker runtime must be available.
