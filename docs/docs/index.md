# Galaxy Classification System

## Introduction
This project was focused on building a custom ResNet model for classifying galaxy images using Galaxy Zoo 2 dataset. This projct not only focus on buidling and training the model, but also on the entire pipeline, including data preprocessing, model architecture design, training strategies, and evaluation metrics. Furthermore those processes are done with MLOps practices in mind, ensuring reproducibility and scalability. Lastly it is served using an API built with FastAPI, allowing for easy integration and deployment.

## Objectives
The main objective of this project was to have a full indepth learning experice of a machine/deep learning architecture, from data preprocessing to model deployment. The project aimed to provide a comprehensive understanding of the entire machine learning pipeline, including data preprocessing, model architecture design, training strategies, and evaluation metrics. Additionally, the project aimed to implement MLOps practices to ensure reproducibility and scalability of the model.


## Key Dependencies

!!! info "Pinned versions"
    These are the versions currently pinned in `requirements.txt`. Keep this table in sync when upgrading.

| Package | Version | Role |
|---|---|---|
| `fastapi` | 0.136.1 | API framework (`api` service) |
| `uvicorn` | 0.47.0 | ASGI server running the API |
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

## Sections

- [Docker Compose](../docs/infra/docker-compose.md) — service breakdown, ports, volumes, and healthchecks for the full stack.