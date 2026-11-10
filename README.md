# Galaxy Classification

[![Deploy Documentation](https://github.com/YubikNakarmi/galaxy-classification/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/YubikNakarmi/galaxy-classification/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)

An end-to-end MLOps pipeline for classifying galaxy morphologies from images, built with PyTorch, FastAPI, and BentoML. The project includes experiment tracking, asynchronous inference, and production observability.

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://www.bentoml.com/"><img src="https://img.shields.io/badge/BentoML-1.4-000000?logo=bentoml&logoColor=white" alt="BentoML"></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-3.13-0194E2?logo=mlflow&logoColor=white" alt="MLflow"></a>
  <a href="https://rich.readthedocs.io/"><img src="https://img.shields.io/badge/Rich-CLI-CA4B3C?logo=python&logoColor=white" alt="Rich"></a>
  <a href="https://click.palletsprojects.com/"><img src="https://img.shields.io/badge/Click-CLI-000000?logo=python&logoColor=white" alt="Click"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  <a href="https://kubernetes.io/"><img src="https://img.shields.io/badge/Kubernetes-ready-326CE5?logo=kubernetes&logoColor=white" alt="Kubernetes"></a>
</p>

> Full documentation is available on the [docs site](https://yubiknakarmi.github.io/galaxy-classification/), generated with MkDocs and deployed automatically on push.

> The trained model is available on [Hugging Face](https://huggingface.co/YubikNakarmi/GalaxyZooCNN).
---
 
## Table of Contents

 
- [Galaxy Classification](#galaxy-classification)
  - [Table of Contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Tech Stack](#tech-stack)
  - [Architecture](#architecture)
  - [Models](#models)
  - [Running](#running)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [File Setup](#file-setup)
    - [Azure Blob Setup (Optional)](#azure-blob-setup-optional)
  - [Database Migrations](#database-migrations)
  - [CLI](#cli)
  - [Training & evaluation](#training--evaluation)
    - [Training on a VM](#training-on-a-vm)
    - [Evaluation](#evaluation)
  - [Testing](#testing)
  - [Azurite (Local Blob Emulator) Testing](#azurite-local-blob-emulator-testing)
  - [Roadmap / Known Improvements](#roadmap--known-improvements)
  - [Sources / References](#sources--references)
  
---


## Motivation

This project combines an interest in astronomy with the goal of building a maintainable,
production-oriented machine-learning system. It is also a practical way to explore modern
ML and DevOps tooling while keeping the complete workflow—from data preparation to
observable inference—in one repository.

## Tech Stack

<table>
  <tr>
    <td align="center"><strong>Machine learning</strong><br>PyTorch<br>torchvision<br>scikit-learn</td>
    <td align="center"><strong>Data & experiments</strong><br>MLflow<br>Azure Blob Storage</td>
    <td align="center"><strong>Serving</strong><br>FastAPI<br>BentoML<br>Celery<br>Redis</td>
  </tr>
  <tr>
    <td align="center"><strong>Observability</strong><br>Prometheus<br>Grafana<br>Loki<br>Grafana Alloy</td>
    <td align="center"><strong>Infrastructure</strong><br>Docker Compose<br>Kubernetes<br>Terraform<br>Azure</td>
    <td align="center"><strong>Developer experience</strong><br>uv<br>Click<br>Rich<br>pytest<br>MkDocs</td>
  </tr>
</table>

## Architecture
![arch](./docs/docs/assets/architecture/galaxy-classification-overview.png)

## Models

- **Default:** Five-class residual CNN.
- **Dirichlet:** A multi-head model with the same backbone as the default model. It predicts voting ratios using a Dirichlet negative log-likelihood loss.

## Running

The project uses a microservice architecture with three main services:

- **Training service:** Trains models with PyTorch and tracks experiments with MLflow.
- **Serving service:** Serves models with FastAPI and BentoML.
- **Inference service:** Handles predictions and exposes metrics for Prometheus and Grafana.

Other supporting services are defined in the Docker Compose configuration.

The `docker-compose.yml` file runs the local services:

```sh
docker compose up
```



## Getting Started

 
### Prerequisites
 
- Python (see `pyproject.toml` / `uv.lock` for the pinned version)
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker & Docker Compose (for local infra)
- An Azure Storage Account (or [Azurite](https://github.com/Azure/Azurite) for local emulation)

**Using uv**
```bash
uv sync --all-extras --frozen # for all deps

uv sync --api # for api deps only
```

**With pip**
```bash
pip install -r requirements.txt
```


### File Setup
 
Create the following directories at the project root and add `.gitkeep` files to each, so they're tracked by git before any real data lands in them:
 
```sh
mkdir -p logs models data
touch logs/.gitkeep models/.gitkeep data/.gitkeep
```
 
### Azure Blob Setup (Optional)
 
1. Create a storage account in Azure (or use **Azurite** for local emulation).
2. Create the required containers within that account.
3. Generate a SAS token or connection string for authentication.
**Required containers:**
 
| Container | Purpose |
|---|---|
| `mlflow` | MLflow tracking backend |
| `galaxy-classification-experiments` | Experiment artifacts |
| `galaxy-classification-logs` | Application/training logs |
 
## Database Migrations
 
Migrations are managed with Alembic:
 
```sh
alembic upgrade head
```

## CLI

The `galax` CLI uses Click for command parsing and Rich for terminal output.

Import a model:

```sh
galax cache import --model-path <path_to_model> --variant <variant_name>
```
## Training & evaluation
 
### Training on a VM
 
For GPU training on a remote VM (e.g., Azure), make sure `nvidia-smi` is available, then install the CUDA-enabled PyTorch build:
 
```sh
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
 
To forward the MLflow UI from the VM to your local machine:
 
```sh
ssh -i .\test-vm_key.pem -L 5000:localhost:5000 azureuser@<vm-ip>
```

Run training script:

```sh
galax pipeline train --variant <variant_name> --evaluate
```
### Evaluation

Evaluation requires a run ID.
 
> **Current model:** ~61.2M parameters (61,187,525).
 
## Testing
 
```sh
# Run a single test module
pytest tests/test_checkpoint_manager.py
 
# Run integration tests
pytest -m integration -v
```
 
## Azurite (Local Blob Emulator) Testing
 
You can validate blob storage integration locally against [Azurite](https://github.com/Azure/Azurite) without touching real Azure resources:
 
```python
from src.utils.azure_utils import AzureBlobStorage
 
blob = AzureBlobStorage(
    storage_connection_string=(
        "DefaultEndpointsProtocol=http;"
        "AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    )
)
 
blob.container_name = "test"
blob.print_info()
```
 
## Roadmap / Known Improvements
 
- **CLI improvements** — improve Rich-based training summaries, pass the UI to child commands, add infrastructure status, and add a `galaxy doctor` command.
- **Model import** — improve the model import workflow.
- **Model export** — support additional export formats.
- **Model cache resilience** — the BentoML model loader cache should fall back to the most recent locally cached "best" model (or pull directly from blob storage) if MLflow is unreachable.
- **Model caching** — cache the loaded model instead of reloading it on every request.
- **Config validation** — validate config paths upfront at startup rather than validating separately per training run.
- **Config management** — evaluate adopting Hydra or OmegaConf instead of the current custom-built parser, which was written under time pressure while focused on the rest of the project.
- **More variants** — add larger models and alternative architectures.
- **Multi-format cache loading** — support formats such as `.onnx`, `.tflite`, and TorchScript in addition to `.pt`.

## Sources / References
- [Draw.io](https://drive.google.com/file/d/1SV51fyxewgoo_lKwKIzrzTUg28_-nhWQ/view?usp=sharing) for building architecture diagrams
- [Galaxy Zoo dataset](https://data.galaxyzoo.org)
- [Galaxy Morphology Classification with Deep Convolutional Neural Networks](https://arxiv.org/abs/1807.10406)
