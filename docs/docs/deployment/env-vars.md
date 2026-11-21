# Environment Variables

Create a repository-root `.env` file for local development. Docker Compose loads
it automatically with `env_file`, and Python tools load it with `python-dotenv`.
Do not commit real credentials or connection strings.

```dotenv
cp .env.template .env
```

## Core project paths

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ENV_PATH` | No | `<repository>/.env` | Path to the environment file loaded by the CLI bootstrap. |
| `ML_YAML_PATH` | No | `configs/ml.yaml` | Path to the ML configuration file. |
| `CORE_YAML_PATH` | No | `configs/core.yaml` | Path to the core application and logging configuration. |
| `BACKEND_CONFIG_PATH` | No | `configs/backend.yaml` | Path to API, database, storage, Celery, and BentoML route settings. |
| `PYTHONPATH` | No | — | Optional Python import path. Use the repository root when running modules outside the activated project environment. |
| `ROOT_DIR` | No | — | Legacy project-root setting listed by the CLI status command; the application now generally derives the root from source paths. |

Example:

```dotenv
ENV_PATH=D:\galaxy_classification-dev\.env
ML_YAML_PATH=D:\galaxy_classification-dev\configs\ml.yaml
CORE_YAML_PATH=D:\galaxy_classification-dev\configs\core.yaml
BACKEND_CONFIG_PATH=D:\galaxy_classification-dev\configs\backend.yaml
PYTHONPATH=D:\galaxy_classification-dev
```

## Azure storage

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | API, training, cloud storage features | — | Azure Blob Storage connection string, or the Azurite connection string for local development. |
| `AZURE_STORAGE_CONNECTION_STRING_LOCAL` | No | — | Optional Azurite/local-storage connection string stored in local `.env` files; application code does not currently read this name directly. |
| `DVC_AZURE_CONTAINER_NAME` | DVC workflows | `dvc-storage` | Azure Blob container used as the DVC remote. |

Use the same container name and connection string consistently for DVC, application
uploads, heatmaps, checkpoints, and MLflow artifacts.

```dotenv
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;..."
DVC_AZURE_CONTAINER_NAME=dvc-storage
```

!!! warning
    Connection strings and storage keys are secrets. Never commit them to Git,
    Dockerfiles, Kubernetes manifests, or documentation examples.

## Database and API backend

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | Production API/database migrations | `sqlite:///app.db` | SQLAlchemy and Alembic database connection URL. |
| `MYSQLURL_REMOTE` | No | — | Legacy MLflow or MySQL connection setting. Prefer `MLFLOW_BACKEND_STORE_URI` for Docker Compose and MLflow. |

Example MySQL URL:

```dotenv
DATABASE_URL=mysql+pymysql://<username>:<password>@<host>:<port>/galaxyclassification
MYSQLURL_REMOTE=mysql+pymysql://<username>:<password>@<host>:<port>/mlflow_db
```

## MLflow

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | No | `http://localhost:5000` | Base URL used by training, evaluation, metrics, artifacts, and health checks. |
| `MLFLOW_TRACKING_USERNAME` | Authenticated MLflow | — | Basic-auth username used by MLflow HTTP requests. |
| `MLFLOW_TRACKING_PASSWORD` | Authenticated MLflow | — | Basic-auth password used by MLflow HTTP requests. |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | No | `0` | Maximum MLflow HTTP request retries. Set internally when missing. |
| `MLFLOW_HTTP_REQUEST_TIMEOUT` | No | `10` | MLflow HTTP request timeout in seconds. Set internally when missing. |
| `MLFLOW_BACKEND_STORE_URI` | MLflow server | — | SQL backend-store URI used by the MLflow container and server command. |
| `MLFLOW_ARTIFACT_ROOT` | MLflow server | — | Artifact root, such as an Azure Blob `wasbs://` URI. |

MLflow server credentials from Docker Compose are supplied by
`MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. Kubernetes manifests
also use the secret keys `username`, `password`, and `host`, and may expose
them to containers as `MLFLOW_USERNAME` or `MLFLOW_PASSWORD`. Application code
should use `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<password>
MLFLOW_HTTP_REQUEST_MAX_RETRIES=0
MLFLOW_HTTP_REQUEST_TIMEOUT=10

MLFLOW_BACKEND_STORE_URI=mysql+pymysql://<username>:<password>@<host>:<port>/mlflow_db
MLFLOW_ARTIFACT_ROOT=wasbs://<container>@<account>.blob.core.windows.net/
```

## BentoML and Celery routes

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `BENTOML_DEFAULT_URL` | No | Backend config or `http://bentoml:3000/classify` | Default model classification endpoint. |
| `BENTOML_HUBBLE_URL` | No | Backend config or `http://bentoml:3000/classify/hubble` | Hubble-variant classification endpoint. |
| `BENTOML_DIRICHLET_URL` | No | Backend config or `http://bentoml:3000/classify/dirichlet` | Dirichlet-variant classification endpoint. |

The routes can alternatively be configured in `configs/backend.yaml` under
`bentoml_routes`.

```dotenv
BENTOML_DEFAULT_URL=http://bentoml:3000/default
BENTOML_HUBBLE_URL=http://bentoml:3000/hubble
BENTOML_DIRICHLET_URL=http://bentoml:3000/dirichlet
```

## Logging and CLI behavior

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LOG_JSON` | No | `false` for CLI bootstrap; Docker examples often set `True` | Emit JSON application logs instead of human-readable text. |
| `NO_LIVE` | No | — | Set to `1` to disable Rich live/animated CLI output. |
| `KAGGLE_KERNEL_RUN_TYPE` | No | — | Kaggle-provided environment marker used to detect Kaggle execution and disable live terminal features. |

```dotenv
LOG_JSON=True
NO_LIVE=1
```

## Grafana

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GF_SECURITY_ADMIN_USER` | Docker Compose Grafana | `admin` recommended | Grafana administrator username. |
| `GF_SECURITY_ADMIN_PASSWORD` | Docker Compose Grafana | — | Grafana administrator password. |

```dotenv
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=<strong-password>
```

## Loading precedence

The project generally loads configuration in this order:

1. Explicit function or CLI arguments.
2. Process environment variables.
3. `.env` values loaded with `python-dotenv`.
4. YAML defaults in `configs/*.yaml`.
5. Built-in Python defaults.

Security-sensitive values should be supplied through environment variables or
secret managers in deployed environments rather than checked-in YAML files.