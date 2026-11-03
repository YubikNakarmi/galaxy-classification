# Model cache commands

The cache group manages local model files and their metadata:

```bash
galax cache --help
```

## `cache import`

Imports a checkpoint into the local model cache and records its metadata.

```bash
galax cache import [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--checkpoint_path TEXT` | **required** | Checkpoint to import. |
| `--model_name TEXT` | **required** | Registered model name. |
| `--variant TEXT` | **required** | Variant name. |
| `--num_classes INTEGER` | **required** | Number of output classes. |
| `--run_id TEXT` | generated | Identifier stored with the checkpoint. |
| `--metric_name TEXT` | `vf1_macro` | Metric used to rank the model. |
| `--metric FLOAT` | — | Metric value when absent from the checkpoint. |
| `--local_cache_dir TEXT` | config | Destination for cached weights. |
| `--metadata_save_file_path TEXT` | config | Metadata JSON path. |

```bash
galax cache import \
  --checkpoint_path models/checkpoint.pt \
  --model_name galaxy-classification-default \
  --variant Default \
  --num_classes 5 \
  --metric 0.82
```

The checkpoint may be a raw state dictionary or a training checkpoint
containing `model_state_dict`.

## `cache export`

```bash
galax cache export [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--model_path TEXT` | `-p` | **required** | Model file to export. |
| `--export_path TEXT` | `-e` | **required** | Destination file or path. |
| `--input_shape TUPLE` | `-i` | `(1, 3, 128, 128)` | Example input shape. |
| `--format [onnx\|torchscript\|torchscript_pt2\|tensorrt]` | `-f` | **required** | Export format. |

```bash
galax cache export \
  -p models/cache/model.pt \
  -e models/export/model.onnx \
  -f onnx \
  -i "(1,3,128,128)"
```

## `cache show-local-cache`

| Option | Default | Description |
| --- | --- | --- |
| `--metadata_path TEXT` | configured metadata path | Metadata JSON to inspect. |
| `--maximum_rows INTEGER` | `5` | Maximum rows requested for display. |

```bash
galax cache show-local-cache --maximum_rows 10
```

## `cache modify`

| Option | Default | Description |
| --- | --- | --- |
| `--run_id TEXT` | — | Run ID to modify. |
| `--promote` | off | Promote the selected model to champion. |

```bash
galax cache modify --run_id RUN_ID --promote
```

## `cache delete`

| Option | Default | Description |
| --- | --- | --- |
| `--run_id TEXT` | — | Run ID to delete. |
| `--model_id TEXT` | — | Model ID to delete. |
| `--variant TEXT` | — | Model variant. |
| `--force` | off | Delete without confirmation. |
| `--local_cache_dir TEXT` | configured cache | Local model directory. |
| `--metadata_save_file_path TEXT` | configured metadata | Metadata JSON path. |

Without `--force`, deletion asks for confirmation.
