## Key Dependencies

!!! info "Pinned versions"
    These are the versions currently pinned in `requirements.txt`. Keep this table in sync when upgrading.

| Package | Version | Role |
|---|---|---|
| `fastapi` | 0.136.1 | API framework (`api` service) |
| `uvicorn` | 0.47.0 | ASGI server running the API |
| `MySQL` | 8.0.33 | Relational database for storing metadata |
| `SQLAlchemy` | 2.0.49| ORM for database interactions |
| `pymysql` | 1.2.0 | MySQL database driver |
| `Alembic`|1.18.4 | Database migrations |
| `bentoml` | 1.4.39 | Model serving (`bentoml` service) |
| `mlflow` | 3.13.0 | Experiment tracking / registry (`mlflow` service) |
| `tensorboard` | 2.20.0 | Training visualization (`tensorboard` service) |
| `torch` | 2.12.0 | Deep learning framework |
| `torchvision` | 0.27.0 | Vision models/transforms |
| `torchmetrics` | 1.9.0 | Model evaluation metrics |
| `numpy` | 2.4.6 | Numerical computing |
| `pandas` | 2.3.3 | Data handling |
| `opencv-python` | 4.13.0.92 | Image processing |
| `dvc` | 3.67.1 | Data/model versioning |
| `azure-storage-blob` | 12.29.0 | Azure Blob Storage client |
| `mkdocs-material` | 9.6.15 | Docs site theme |
|`loki`|2.9.0|Log aggregation|
|`promtail`|2.9.0|Log aggregation agent|

## Other Dependancies

!!! info "Other dependencies"
    Other dependencies that are needes for `bentoml` and `api` are listed in their respective directories. Currently listed are 
    - `requirements.bentoml.txt` — dependencies for the `bentoml_api/` service.
    - `requirements.api.txt` — dependencies for the `api/` service.

- [Docker Compose](../docs/infra/docker-compose.md) — service breakdown, ports, volumes, and healthchecks for the full stack.


