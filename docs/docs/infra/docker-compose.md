# Docker Compose Stack Documentation

This document describes the services defined in `docker-compose.yml` for the galaxy classification project: the API layer, model serving, experiment tracking, and observability stack.


## Overview

| Service | Image / Build | Host Port(s) | Purpose |
|---|---|---|---|
| `api` | built from `infra/docker/Dockerfile.api` | 8001 → 8000 | FastAPI application layer |
| `bentoml` | `galaxy_classification:v0.6.0` | 3000 | Model serving (BentoML bento) |
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.13.0-full` | 5000 | Experiment tracking / model registry |
| `grafana` | `grafana/grafana:13.1.0` | 3001 → 3000 | Metrics dashboards |
| `prometheus` | `prom/prometheus:v3.13.1` | 9090 | Metrics collection |
| `redis` | `redis:7-alpine` | 6379 | Cache / queue backend |
| `azureite` | `mcr.microsoft.com/azure-storage/azurite` | 10000-10002 | Local Azure Blob Storage emulator |
| `tensorboard` | `tensorflow/tensorflow:latest` | 6006 | Training run visualization |

Start the full stack with:

```bash
docker compose up -d
```

---

## `api`

FastAPI service, built locally from `infra/docker/Dockerfile.api`.

- **Container name:** `api`
- **Build context:** `.` (repo root), Dockerfile at `infra/docker/Dockerfile.api`
- **Port mapping:** `8001:8000` (host `8001` → container `8000`)
- **Command:** `uvicorn api/main:app --host 0.0.0.0`
- **Env file:** `.env`
- **Extra environment:**
  - `MLFLOW_TRACKING_URI=http://mlflow:5000`

**Volumes:**

| Host path | Container path | Purpose |
|---|---|---|
| `./api` | `/api` | Application source |
| `./src` | `/src` | Shared source code |
| `./configs` | `/configs` | Configuration files |
| `./logs/api` | `/logs/system` | API logs |


---

## `grafana`

Dashboarding UI for metrics stored in Prometheus.

- **Image:** `grafana/grafana:13.1.0`
- **Container name:** `grafana`
- **Port mapping:** `3001:3000` (moved off `3000` to avoid clashing with BentoML)
- **Env file:** `.env`
- **Depends on:** `prometheus` (waits for `service_healthy`, not just container start)
- **Volume:** `grafana_data:/var/lib/grafana` (named volume, persists dashboards/users)
- **Healthcheck:** `curl -f http://localhost:3000`, every 10s, 50 retries, 30s start period

---

## `prometheus`

Metrics scraper/storage.

- **Image:** `prom/prometheus:v3.13.1`
- **Container name:** `prometheus`
- **Restart policy:** `unless-stopped`
- **Port mapping:** `9090:9090`

**Volumes:**

| Host path | Container path | Purpose |
|---|---|---|
| `./prometheus.yml` | `/etc/prometheus/prometheus.yml` | Scrape config |
| `prometheus_data` (named) | `/prometheus` | TSDB storage |

**Healthcheck:** `curl -f http://localhost:9090`, every 10s, 50 retries, 30s start period

---

## `redis`

Cache / broker backend, persisted with AOF.

- **Image:** `redis:7-alpine`
- **Command:** `redis-server --appendonly yes`
- **Port mapping:** `6379:6379`
- **Volume:** `redis-data:/data`
- **Healthcheck:** `redis-cli ping`, every 10s, 50 retries, 30s start period
- **Restart policy:** `always`

!!! note
    This service has no `container_name` set (all others do), so Compose will auto-name it (e.g. `<project>_redis_1` / `<project>-redis-1`).

---

## `bentoml`

Serves the packaged BentoML bento (`galaxy_classification:v0.6.0`). It is build from  `bentoml_api` using the command 

```bash
bentoml build --version v0.6.0 --containerize 
```
!!! note
     Can use other versions of the bento just make sure to sync the tag and container in the compose. 

- **Image:** `galaxy_classification:v0.6.0` — built via `bentoml build`, not built by this compose file
- **Container name:** `bentoml`
- **Port mapping:** `3000:3000`
- **Env file:** `.env`
- **Depends on:** `mlflow` (waits for `service_healthy`, not just container start)
- **Healthcheck:** `curl -f http://localhost:3000`, every 10s, 50 retries, 30s start period

**Extra environment:**
- `MLFLOW_TRACKING_URI=http://mlflow:5000`
- `AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING}`
- `LOG_JSON=${LOG_JSON}`

**Volumes:**

| Host path | Container path | Purpose |
|---|---|---|
| `./bentoml` | `/bentoml` | BentoML runtime data |
| `./bentoml_api` | `/home/bentoml/bento/src/bentoml_api` | API source override |
| `./src` | `/home/bentoml/bento/src/src` | Editable shared source (no rebuild needed) |
| `./logs/bentoml` | `/home/bentoml/bento/src/logs/system/` | Service logs |
| `./models/cached` | `/home/bentoml/bento/src/models/cached` | Persisted cached models |
| `./configs` | `/home/bentoml/bento/src/configs` | Configuration files |

---

## `mlflow`

Experiment tracking server and model registry, backed by a remote MySQL store and remote artifact root.

- **Image:** `ghcr.io/mlflow/mlflow:v3.13.0-full`
- **Container name:** `mlflow`
- **Port mapping:** `5000:5000`
- **Env file:** `.env`
- **Environment:**
  - `MYSQLURL_REMOTE=${MYSQLURL_REMOTE}` — backend store URI (MySQL)
  - `MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT}` — artifact storage location (e.g. Azure Blob / Azurite)
- **Command:**
  ```
  mlflow server
    --backend-store-uri ${MYSQLURL_REMOTE}
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT}
    --host 0.0.0.0
    --port 5000
    --allowed-hosts "*"
  ```
- **Healthcheck:** Python `urllib.request` hitting `http://localhost:5000/health`, every 10s, 5 retries, 10s start period

!!! note
    The `mlflow db upgrade ...` line is commented out inside the `command: >` block — DB migrations are **not** run automatically on startup. Run them manually if the schema changes.

---

## `azureite`

Local emulator for Azure Blob/Queue/Table Storage, used as a stand-in for real Azure Storage (referenced by `AZURE_STORAGE_CONNECTION_STRING` and `MLFLOW_ARTIFACT_ROOT`).

- **Image:** `mcr.microsoft.com/azure-storage/azurite`
- **Container name:** `azureite`
- **Ports:**
  - `10000` — Blob service
  - `10001` — Queue service
  - `10002` — Table service
- **Command:** `azurite --blobHost 0.0.0.0 --blobPort 10000 --skipApiVersionCheck`
- **Volumes:** none defined — **data is not persisted** across container recreation.

---

## `tensorboard`

Visualizes training runs/logs.

- **Image:** `tensorflow/tensorflow:latest`
- **Container name:** `tensorboard`
- **Port mapping:** `6006:6006`
- **Volume:** `./data/ml/tensorboard/experiment_1:/tensorboard`
- **Command:** `tensorboard --logdir=/tensorboard --host=0.0.0.0`

!!! note
    Only `experiment_1` is mounted — additional experiment directories won't be visible unless the volume path is changed or a parent directory is mounted instead.

---

## Named Volumes

| Volume | Used by | Purpose |
|---|---|---|
| `prometheus_data` | `prometheus` | Time-series metrics storage |
| `redis-data` | `redis` | AOF persistence |
| `grafana_data` | `grafana` | Dashboards, users, plugins |

---

## Required Environment Variables (`.env`)

| Variable | Used by | Description |
|---|---|---|
| `MYSQLURL_REMOTE` | `mlflow` | MySQL connection string for the MLflow backend store |
| `MLFLOW_ARTIFACT_ROOT` | `mlflow` | Root URI for MLflow artifact storage |
| `AZURE_STORAGE_CONNECTION_STRING` | `bentoml` | Connection string for Azure Blob Storage (or Azurite) |
| `LOG_JSON` | `bentoml` | Toggles JSON-formatted logging |

All services also load `.env` directly via `env_file`, so any other project secrets/config should live there as well.

---


## Known Issues / Suggested Fixes

!!! danger "Review before deploying"
    1. **`azureite`** has no volume, so emulated storage is lost on container recreation.
    2. **`bentoml`** image must be build using the bento from `bentoml.yaml`.

!!! success "Resolved"
    - `prometheus` now has a healthcheck.
    - `grafana` and `bentoml` now use `condition: service_healthy` in `depends_on`, so they wait for `prometheus`/`mlflow` to actually be ready, not just started.