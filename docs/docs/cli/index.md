# Command-line interface

The project installs the `galax` command when the package is installed:

```bash
uv sync
uv run galax --help
```

Or install the package in an activated virtual environment:

```bash
pip install -e .
galax --help
```

!!! tip
    Run `galax COMMAND --help` for command-specific options. Add `-v`, `-vv`,
    or `-vvv` immediately after `galax` for more diagnostic output.

## Command tree

| Command | Purpose |
| --- | --- |
| `galax status` | Display project configuration and service status. |
| `galax log show` | Read and filter application or ML logs. |
| `galax pipeline ...` | Preprocess data, train, evaluate, and load MLflow metrics. |
| `galax cache ...` | Import, inspect, export, modify, or delete local models. |
| `galax setup ...` | Create project directories. |
| `galax cloud ...` | Check Azure Blob Storage or upload files. |
| `galax bentoml ...` | Build and containerize the BentoML service. |

## Documentation map

- [Global usage and logging](global.md)
- [ML pipeline commands](pipeline.md)
- [Model cache commands](cache.md)
- [Setup, cloud, and BentoML commands](integrations.md)
- [Training and evaluation workflow](train_and_eval.md)

Every command and command group accepts `-h`/`--help`.
