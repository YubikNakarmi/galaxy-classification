# Galaxy Classification System

## Movitation
The reason for me working of project is due to multitude of factors. Firstly, I was always interested in astronomy and Neural networks and I wanted to buiild something regarding Astronomy. Then there also the motivation to build something better than my lat project [AQI_predictor](https://github.com/YubikNakarmi/Aqi_predictor), which is pretty messy. I wanted to contiune to learn to build better systems wich more cleaner and maintainable code with also better organizations and potability. With as miminal use of AI as possible, except is somethign like comments, docstrings or documentation .Lastly, I also wanted to learn new tools for DevOPS, Mlops, and ML in general I leared tools like:

- Pytorch
- FastAPI
- BentoML
- Promethus
- Grafana
   and more


## Introduction
This project was focused on building a custom ResNet model for classifying galaxy images using Galaxy Zoo 2 dataset[@Willett_2013]. This projct not only focus on buidling and training the model, but also on the entire pipeline, including data preprocessing, model architecture design, training strategies, and evaluation metrics. Furthermore those processes are done with MLOps practices in mind, ensuring reproducibility and scalability. 

Lastly it is served using an API built with FastAPI, allowing for easy integration and deployment. This projects expands on [@Zhu_2019] general architecture of ResNet and implements a whole end to end pipeline for galaxy classification. The project is structured as a microservices architecture, with each service responsible for a specific aspect of the project. Each service is deployed through **containers** which can be scaled independently using tools like Docker Swarm or Kubernetes.

## Objectives
The main objective of this project was to have a full indepth learning experice of a machine/deep learning architecture, from data preprocessing to model deployment. The project aimed to provide a comprehensive understanding of the entire machine learning pipeline, including data preprocessing, model architecture design, training strategies, and evaluation metrics. Additionally, the project aimed to implement MLOps practices to ensure reproducibility and scalability of the model.

## Microservices Architecture
The project is structured as a microservices architecture, with each service responsible for a specific aspect of the project. Each service is deployed through **containers** which can be scaled independently using tools like Docker Swarm or Kubernetes. The services are as follows:

- `api` — FastAPI service that serves the model and provides an endpoint for predictions.
- `bentoml` — BentoML service that handles model serving and deployment.
- `mlflow` — MLflow service that tracks experiments, logs metrics, and manages model versions.

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


/references