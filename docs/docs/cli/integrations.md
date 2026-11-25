# Setup, cloud, and BentoML commands

## Setup

`setup` currently exposes `mkdirs`:

```bash
galax setup --help
galax setup mkdirs [--extra_dirs TEXT]
```

| Option | Description |
| --- | --- |
| `--extra_dirs TEXT` | Comma-separated list of additional directories to create. |

```bash
galax setup mkdirs --extra_dirs "reports,exports"
```

## Azure cloud storage

Cloud commands require `AZURE_STORAGE_CONNECTION_STRING` in the environment or
`.env` file.

### `cloud status`

```bash
galax cloud status
```

Displays Azure Storage account and container information. It has no
project-specific options.

### `cloud upload-blob`

```bash
galax cloud upload-blob [OPTIONS]
```

| Option | Short form | Required | Description |
| --- | --- | --- | --- |
| `--container_name TEXT` | — | no | Azure container; uses configured default when omitted. |
| `--blob_name TEXT` | `-b` | yes | Destination blob name. |
| `--file_path TEXT` | `-f` | yes | Local file to upload. |

```bash
galax cloud upload-blob \
  --container_name experiments \
  --blob_name reports/run.json \
  --file_path reports/run.json
```

## BentoML

`bentoml` exposes the `containerize` command:

```bash
galax bentoml --help
galax bentoml containerize [OPTIONS]
```

| Option | Short form | Default | Description |
| --- | --- | --- | --- |
| `--bentofile-path TEXT` | `-f` | `bentofile.yaml` | BentoML configuration file or directory. |
| `--tag TEXT` | `-t` | `latest` | Docker image version tag. |

```bash
galax bentoml containerize \
  --bentofile-path bentofile.yaml \
  --tag latest
```

The command invokes BentoML and Docker locally, so both executables and a
working Docker runtime must be available.
